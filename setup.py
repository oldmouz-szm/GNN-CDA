from setuptools import setup, find_packages

setup(
    name="gnn-cda-diagnosis",
    version="1.0.0",
    description="GNN-Guided Conflict-Directed A* Search for Circuit Diagnosis",
    author="Your Name",
    packages=find_packages(),
    py_modules=[
        "astar_search",
        "circuit_simulator",
        "circuit_gnn_converter",
        "train_gnn",
        "generate_dataset"
    ],
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "networkx>=3.0",
        "numpy"
    ],
    entry_points={
        'console_scripts': [
            'gnn-cda=astar_search:main',
        ],
    },
    python_requires='>=3.8',
)
