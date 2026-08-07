"""
Example script for federated learning in BioOncoGraph.
"""
import argparse
import logging

from biooncograph.federated import FederatedClient, FederatedServer, FedAvgStrategy
from biooncograph.models import GNNClassifier
from biooncograph.datasets import LocalDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def start_federated_node(node_type: str, data_path: str = "") -> None:
    """Starts a federated learning node (client or server).

    Args:
        node_type: Either 'client' or 'server'.
        data_path: Path to local data for the client.
    """
    if node_type == "server":
        logger.info("Initializing Federated Server with FedAvg strategy")
        strategy = FedAvgStrategy(min_clients=2)
        server = FederatedServer(strategy=strategy, port=50051)
        server.start()
    elif node_type == "client":
        if not data_path:
            raise ValueError("data_path must be provided for client node")
            
        logger.info(f"Initializing Federated Client with local data from {data_path}")
        dataset = LocalDataset(data_path=data_path)
        model = GNNClassifier(hidden_dim=128, num_classes=dataset.num_classes)
        
        client = FederatedClient(server_address="localhost:50051", model=model, dataset=dataset)
        client.start()
    else:
        logger.error(f"Unknown node type: {node_type}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Node")
    parser.add_argument("--node_type", type=str, choices=["client", "server"], required=True)
    parser.add_argument("--data_path", type=str, default="", help="Path to local data (client only)")
    args = parser.parse_args()
    start_federated_node(args.node_type, args.data_path)
