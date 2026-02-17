import torch
import torch.nn as nn
import time
from models import MLP, SimpleCNN, TabularTransformer
from data_loader import get_data
from sklearn.metrics import accuracy_score

def run_9_experiments():
    datasets = ["Adult", "CIFAR100", "PCam"]
    archs = ["MLP", "CNN", "Transformer"]
    final_results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for d_name in datasets:
        for a_name in archs:
            print(f"\n" + "="*50)
            print(f"EXPERIMENT: {d_name} using {a_name}")
            print("="*50)
            
            train_loader, test_loader, in_dim, in_chan, n_class, res = get_data(d_name, 128)

            if a_name == "MLP": model = MLP(in_dim, n_class).to(device)
            elif a_name == "CNN": model = SimpleCNN(in_chan, n_class, res).to(device)
            else: model = TabularTransformer(in_dim, n_class).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            start_t = time.time()
            # Running for 5 epochs 
            for epoch in range(1, 6):
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    if a_name == "CNN" and d_name == "Adult": x = x.view(-1, 1, 4, 4)
                    
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
                
                # Calculate Epoch Metrics
                epoch_acc = 100 * correct / total
                avg_loss = running_loss / len(train_loader)
                print(f"Epoch [{epoch}/5] - Loss: {avg_loss:.4f} - Training Acc: {epoch_acc:.2f}%")
            
            # Final Evaluation on Test Set
            model.eval()
            all_preds, all_y = [], []
            with torch.no_grad():
                for x, y in test_loader:
                    if a_name == "CNN" and d_name == "Adult": x = x.view(-1, 1, 4, 4)
                    out = model(x.to(device))
                    all_preds.append(torch.argmax(out, dim=1).cpu())
                    all_y.append(y)
            
            final_acc = accuracy_score(torch.cat(all_y), torch.cat(all_preds))
            duration = time.time() - start_t
            
            print(f"--- Finished {a_name} on {d_name} | Test Acc: {final_acc:.4f} | Time: {duration:.1f}s ---")
            final_results.append({"Dataset": d_name, "Arch": a_name, "Acc": final_acc, "Time": duration})

    return final_results

if __name__ == "__main__":
    #capture the return value
    results_data = run_9_experiments()

    # Final Table Printout
    print("\n" + "!"*40 + "\nFINAL SUMMARY TABLE\n" + "!"*40)
    for r in results_data:
        print(f"{r['Dataset']:<10} | {r['Arch']:<12} | Acc: {r['Acc']:.4f} | Time: {r['Time']:.1f}s")

    # Generate the Charts
    try:
        from plotting_utils import save_benchmark_plots
        save_benchmark_plots(results_data)
        print("\n[SUCCESS] Plots saved to directory.")
    except ImportError:
        print("\n[SKIP] plotting_utils.py not found. Charts not generated.")
