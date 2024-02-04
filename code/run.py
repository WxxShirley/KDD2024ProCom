from copy import deepcopy
import data
import metrics
import pretrain
import utils
import argparse
import torch
import model
import numpy as np
import time
import os
import math
from torch_geometric.utils import k_hop_subgraph

if __name__ == "__main__":
    print('= ' * 20)
    print('## Starting Time:', utils.get_cur_time(), flush=True)

    parser = argparse.ArgumentParser(description="ComGPPT")

    # Dataset choices
    #  ['amazon', 'dblp', 'lj', 'facebook', 'twitter']
    parser.add_argument("--dataset", type=str, default="amazon_small")
    parser.add_argument("--seeds", type=list, default=[2023, 12345, 42, 6666, 0, 1999, 2020, 12345678, 9999, 1, 2, 3])
    parser.add_argument("--device", type=str, default="cuda:1")

    # training related
    parser.add_argument("--pretrain_method", type=str, default="ComGPPT")
    parser.add_argument("--pretrain_epoch", type=int, default=30)
    parser.add_argument("--prompt_epoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)

    # task related
    #  for pretrain
    parser.add_argument("--from_scratch", type=int, default=1)
    parser.add_argument("--node_scale", type=float, default=1.0)
    parser.add_argument("--subg_scale", type=float, default=1.0)

    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--max_subgraph_size", type=int, default=20)
    parser.add_argument("--num_shot", type=int, default=10)
    parser.add_argument("--num_pred", type=int, default=1000)
    parser.add_argument("--run_times", type=int, default=10)
    parser.add_argument("--generate_k", type=int, default=2)

    # GNN related
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--gnn_type", type=str, default="GCN")

    # Prompt related
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()
    # for facebook, only predict 200 communities
    if args.dataset == "facebook":
        args.num_pred = 200
    elif args.dataset in ["dblp", "amazon", "twitter"]:
        args.num_pred = 5000

    if args.dataset == "twitter":
        args.threshold = 0.1

    print(args)
    print("\n")

    ##########################################################
    ################### Step 1 Load Data #####################
    ##########################################################
    num_node, num_edge, num_community, graph_data, nx_graph, communities = data.prepare_data(args.dataset)
    print(f"Finish loading data: {graph_data}\n")

    ##########################################################
    ################### Step 2 Pre-train GNN #################
    ##########################################################
    input_dim = graph_data.x.size(1)
    device = torch.device(args.device)
    print("Perform pre-training ... ")
    print(f"GNN Configuration gnn_type({args.gnn_type}), num_layer({args.n_layers}), hidden_dim({args.hidden_dim})")

    utils.set_seed(args.seeds[0])

    pretrain_model = pretrain.PreTrain(dataset=args.dataset,
                                       gnn_type=args.gnn_type,
                                       input_dim=input_dim,
                                       hidden_dim=args.hidden_dim,
                                       num_layers=args.n_layers,
                                       device=device)
    print(pretrain_model.gnn)
    num_pretrain_param = sum(p.numel() for p in pretrain_model.gnn.parameters())
    print(f"[Parameters] Number of parameters in GNN {num_pretrain_param}")

    pretrain_file_path = f"pretrain_models/{args.dataset}_{args.node_scale}_{args.subg_scale}_model.pt"
    if not args.from_scratch and os.path.exists(pretrain_file_path):
        pretrain_model.gnn.load_state_dict(torch.load(pretrain_file_path))
        print(f"Loading PRETRAIN-GNN file from {pretrain_file_path} !\n")
    else:
        if args.pretrain_method == "ComGPPT":
            print("Pretrain with ComGPPT proposed community-centric SSL Loss ... ")
            pretrain_model.train(graph_data,
                                 batch_size=args.batch_size,
                                 lr=args.lr,
                                 epochs=args.pretrain_epoch,
                                 subg_max_size=args.max_subgraph_size,
                                 num_hop=args.k,
                                 node_scale=args.node_scale,
                                 subg_scale=args.subg_scale)
            if not args.from_scratch:
                torch.save(pretrain_model.gnn.state_dict(), pretrain_file_path)
        print(f"Pretrain Finish!\n")

    ##########################################################
    ####### Step 3 (Pre)-processing [NodeEmbed, KEgoNet] #####
    ##########################################################
    all_node_emb = pretrain_model.generate_all_node_emb(pretrain_model.gnn, graph_data.to(device))
    all_node_emb = all_node_emb.detach()
    print("Pre-processing for K-EGO-NET extraction")
    node2ego_mapping = []
    # we also extract fixed k-ego net to avoid exhaustive re-computation
    if os.path.exists(f"../data/{args.dataset}/{args.generate_k}-ego.txt"):
        with open(f"../data/{args.dataset}/{args.generate_k}-ego.txt", 'r') as file:
            for line in file.readlines():
                content = line.split(" ")[1:]
                node2ego_mapping.append([int(node) for node in content])
    else:
        for node in range(num_node):
            node_k_ego, _, _, _ = k_hop_subgraph(node, num_hops=args.generate_k, edge_index=graph_data.edge_index,
                                                 num_nodes=num_node)
            node_k_ego = node_k_ego.detach().cpu().numpy().tolist()
            node2ego_mapping.append(node_k_ego)
            if node % 5000 == 0:
                print(f"***pre-processing {node} finish")
    print("Pre-preocessing Finish!\n")

    # for every sample, number of matched communities
    num_single_match = int(args.num_pred / args.num_shot)
    all_scores = []
    for j in range(args.run_times):
        ##########################################################
        ################## Step 4 Prompt Tuning ##################
        ##########################################################
        print(f"Times {j}")
        utils.set_seed(args.seeds[j])

        prompt_model = model.PromptLinearNet(args.hidden_dim, threshold=args.threshold).to(device)
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(prompt_model.parameters(), lr=args.lr, weight_decay=0.00001)

        num_prompt_param = sum(p.numel() for p in prompt_model.parameters())
        print(f"[Parameters] Number of parameters in Prompt {num_prompt_param}")

        # Step 4.1 - Split Communities into Train / Test
        random_idx = list(range(num_community))
        np.random.shuffle(random_idx)
        print(random_idx[:args.num_shot])
        train_comms, test_comms = [communities[idx] for idx in random_idx[:args.num_shot]], [communities[idx] for idx in
                                                                                             random_idx[args.num_shot:]]
        train_com_emb = pretrain_model.generate_target_community_emb(pretrain_model.gnn,
                                                                     train_comms,
                                                                     graph_data).detach()

        for epoch in range(args.prompt_epoch):
            prompt_model.train()
            optimizer.zero_grad()
            st_time = time.time()

            all_central_nodes, all_ego_nodes, all_labels = torch.FloatTensor().to(device), torch.FloatTensor().to(
                device), torch.FloatTensor().to(device)

            # Step 4.2 - Prepare Training data for Prompt Tuning
            for community in train_comms:
                central_node, k_ego, label = utils.generate_prompt_tuning_data(community, graph_data, nx_graph,
                                                                               args.generate_k)
                # print(central_node, k_ego, label)
                central_node_emb = all_node_emb[central_node, :]
                ego_node_emb = all_node_emb[k_ego, :]

                all_labels = torch.cat((all_labels, label.to(device)), dim=0)
                all_central_nodes = torch.cat((all_central_nodes, central_node_emb), dim=0)
                all_ego_nodes = torch.cat((all_ego_nodes, ego_node_emb), dim=0)

            pred_logits = prompt_model(all_ego_nodes, all_central_nodes)

            pt_loss = loss_fn(pred_logits, all_labels)
            pt_loss.backward()
            optimizer.step()
            print("***epoch: {:04d} | PROMPT TUNING train_loss: {:.5f} | cost time {:.3}s".format(epoch, pt_loss,
                                                                                                  time.time() - st_time))
        print(f"Prompt Tuning Finish!\n")

        ##########################################################
        ################## Step 5 Candidate Filtering ############
        ##########################################################
        prompt_model.eval()
        candidate_comms, raw_candidate_comms = [], []

        st_time = time.time()
        for node in range(num_node):
            # for each node, extract its k-ego net and feed to prompt model, return a refined candidate structure
            node_k_ego = node2ego_mapping[node]
            assert node in node_k_ego

            final_pos = prompt_model.make_prediction(all_node_emb[node_k_ego, :],
                                                     all_node_emb[[node] * len(node_k_ego), :])

            if len(final_pos) > 0:
                candidate = [node_k_ego[idx] for idx in final_pos]

                if node not in candidate:
                    candidate = [node] + candidate

                candidate_comms.append(candidate)
                raw_candidate_comms.append(deepcopy(node_k_ego))
        print(f"Finish Candidate Filtering, Cost Time {time.time() - st_time:.5}s!\n")

        ##########################################################
        ######### Step 6 Matching for final Predictions ##########
        ##########################################################
        candidate_com_embeds = None

        # you need to set smaller batch_size if GPU memory is limited
        batch_size = args.batch_size if args.dataset not in ["lj", "twitter", "dblp"] else 32
        num_batch = math.ceil(len(candidate_comms) / batch_size)
        st_time = time.time()
        for i in range(num_batch):
            start, end = i * batch_size, min((i + 1) * batch_size, len(candidate_comms))

            tmp_emb = pretrain_model.generate_target_community_emb(pretrain_model.gnn, candidate_comms[start:end],
                                                                   graph_data)
            tmp_emb = tmp_emb.detach().cpu().numpy()
            if candidate_com_embeds is None:
                candidate_com_embeds = tmp_emb
            else:
                candidate_com_embeds = np.vstack((candidate_com_embeds, tmp_emb))
        print(f"Finish Candidate Embedding Computation, Cost Time {time.time() - st_time:.5}s!\n")

        train_com_emb = train_com_emb.detach().cpu().numpy()
        pred_comms = []
        # st_time = time.time()
        for i in range(args.num_shot):
            query = train_com_emb[i, :]
            distance = np.sqrt(np.sum(np.asarray(query - candidate_com_embeds) ** 2, axis=1))

            sort_dic = list(np.argsort(distance))

            length = 0

            for idx in sort_dic:
                if length >= num_single_match:
                    break

                neighs = candidate_comms[idx]
                if neighs not in pred_comms:
                    pred_comms.append(neighs)
                    length += 1

        f1, jaccard = metrics.eval_scores(pred_comms, test_comms, tmp_print=True)
        all_scores.append([f1, jaccard])
        utils.pred_community_analysis(pred_comms)
        print("\n")
        del prompt_model

    avg_scores = np.mean(np.array(all_scores), axis=0)
    std_scores = np.std(np.array(all_scores), axis=0)
    print(
        f"Overall F1 {avg_scores[0]:.4f}+-{std_scores[0]:.5f}, Overall Jaccard {avg_scores[1]:.4f}+-{std_scores[1]:.5f}")
    print('\n## Finishing Time:', utils.get_cur_time(), flush=True)
    print('= ' * 20)
    print("Done!")
