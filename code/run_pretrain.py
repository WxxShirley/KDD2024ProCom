import data
import metrics
import pretrain
import utils
import argparse
import torch
import numpy as np
from torch_geometric.utils import k_hop_subgraph

if __name__ == "__main__":
    print('= ' * 20)
    print('## Starting Time:', utils.get_cur_time(), flush=True)

    parser = argparse.ArgumentParser(description="ComGPPT")

    # Dataset choices
    #  ['amazon', 'dblp', 'lj', 'facebook', 'twitter']
    parser.add_argument("--dataset", type=str, default="amazon")
    parser.add_argument("--seeds", type=list, default=[2023, 12345, 42, 6666, 0, 1999, 2020, 12345678, 9999, 1, 2, 3])
    parser.add_argument("--device", type=str, default="cuda:0")

    # training related
    parser.add_argument("--pretrain_epoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)

    # task related
    #  for pretrain
    parser.add_argument("--no_pretrain", type=bool, default=False)  # if ture, perform no-pretraining
    parser.add_argument("--pretrain_method", type=str, default="ComGPPT")

    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--max_subgraph_size", type=int, default=20)
    parser.add_argument("--num_shot", type=int, default=10)
    parser.add_argument("--num_pred", type=int, default=1000)
    parser.add_argument("--run_times", type=int, default=10)

    parser.add_argument("--generate_k", type=int, default=2)
    parser.add_argument("--generate_subgraph_size", type=int, default=20)

    # GNN related
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--gnn_type", type=str, default="GCN")

    args = parser.parse_args()

    # for facebook, only predict 200 communities / for DBLP/Amazon/Twitter, predict 5000 communities
    if args.dataset == "facebook":
        args.num_pred = 200
    elif args.dataset in ["dblp", "amazon", "twitter"]:
        args.num_pred = 5000

    print(args)
    print("\n")

    # STEP1 - Load data
    num_node, num_edge, num_community, graph_data, nx_graph, communities = data.prepare_data(args.dataset)
    print(f"Finish loading data: {graph_data}\n")

    # STEP2 - Pretrain GNN
    #    Core Parameters: gnn_type, n_layer, hidden_dim
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
                                       device=device,
                                       )
    print(pretrain_model.gnn)

    if not args.no_pretrain:
        if args.pretrain_method == "ComGPPT":
            print("Pretrain with ComGPPT proposed community-centric SSL Loss ... ")
            pretrain_model.train(graph_data,
                                 batch_size=args.batch_size,
                                 lr=args.lr,
                                 epochs=args.pretrain_epoch,
                                 subg_max_size=args.max_subgraph_size,
                                 num_hop=args.k,
                                 )

    all_candidates_emb = pretrain_model.generate_all_candidate_community_emb(
        pretrain_model.gnn,
        graph_data.to(torch.device("cpu")),
        batch_size=4096,
        k=args.generate_k,
        max_size=args.generate_subgraph_size
    )
    all_candidates_emb = all_candidates_emb.detach().cpu().numpy()

    num_single_match = int(args.num_pred / args.num_shot)
    all_scores = []
    # make predictions right after pre-trained GNN models
    for j in range(args.run_times):
        print(f"Times {j}")
        utils.set_seed(args.seeds[j])

        # split communities into train / test
        random_idx = list(range(num_community))
        np.random.shuffle(random_idx)
        train_comms = [communities[idx] for idx in random_idx[:args.num_shot]]
        test_comms = [communities[idx] for idx in random_idx[args.num_shot:]]
        pred_comms, seeds = [], []
        # check for a fair comparison
        # print(random_idx[:args.num_shot])

        # compute training communities' embed
        train_emb = pretrain_model.generate_target_community_emb(
            pretrain_model.gnn,
            train_comms,
            graph_data
        )
        train_emb = train_emb.detach().cpu().numpy()

        # matching
        for i in range(args.num_shot):
            query = train_emb[i, :]
            distance = np.sqrt(np.sum(np.asarray(query - all_candidates_emb) ** 2, axis=1))

            sort_dic = list(np.argsort(distance))

            length = 0
            for seed_node in sort_dic:
                if length >= num_single_match:
                    break

                if int(seed_node) in seeds:
                    continue

                neighs, _, _, _ = k_hop_subgraph(int(seed_node), num_hops=args.generate_k,
                                                 edge_index=graph_data.edge_index, num_nodes=num_node)
                neighs = sorted(neighs.detach().cpu().numpy().tolist())

                if neighs not in pred_comms:
                    pred_comms.append(neighs)
                    length += 1
                    seeds.append(int(seed_node))

        f1, jaccard = metrics.eval_scores(pred_comms, test_comms, tmp_print=True)
        all_scores.append([f1, jaccard])
        utils.pred_community_analysis(pred_comms)
        print("\n")

    avg_scores = np.mean(np.array(all_scores), axis=0)
    # print(all_scores, avg_scores)
    print(f"Overall F1 {avg_scores[0]:.4f}, Overall Jaccard {avg_scores[1]:.4f}")
    print('\n## Finishing Time:', utils.get_cur_time(), flush=True)
    print('= ' * 20)
    print("Done!")
