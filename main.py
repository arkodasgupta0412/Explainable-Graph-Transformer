import torch
from models.graph_transformer import GraphTransformerNet
from training.train_eval import train, evaluate
from data.datasets import prepare_data


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datasets = ['Cora', 'CiteSeer', 'PubMed']

    for dataset_name in datasets:
        print("=" * 50)
        print(f"Processing {dataset_name}")
        print("=" * 50)

        data = prepare_data(dataset_name).to(device)
        model = GraphTransformerNet(
            in_dim=data.x.size(1),
            hidden_dim=64,
            num_layers=3,
            num_heads=4,
            n_classes=int(data.y.max().item()) + 1,
            dropout=0.3
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)

        best_val_acc = 0
        for epoch in range(1, 501):
            train_loss = train(model, data, optimizer, device)
            val_acc, _ = evaluate(model, data, data.val_mask)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'best_model_{dataset_name}.pt')

            if epoch % 10 == 0:
                train_acc, _ = evaluate(model, data, data.train_mask)
                print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        model.load_state_dict(torch.load(f'best_model_{dataset_name}.pt'))
        test_acc, test_mae = evaluate(model, data, data.test_mask)
        print(f"- Final Test Acc: {test_acc:.4f}, Test MAE: {test_mae:.4f}")
        print("=" * 50)


if __name__ == "__main__":
    main()
