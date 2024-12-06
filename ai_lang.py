#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

env = {
    "commands": {},
    "vars": {},
}

def parse_keywords(args):
    """Parse a list of arguments into positional and keyword arguments."""
    pos_args = []
    kw_args = {}
    i = 0
    while i < len(args):
        if isinstance(args[i], str) and args[i].startswith(":"):
            kw_args[args[i][1:]] = args[i + 1]
            i += 2
        else:
            pos_args.append(args[i])
            i += 1
    return pos_args, kw_args

def debug_define(args):
    """Define a variable by evaluating its value expression."""
    var_name, value_expr = args
    value = evaluate(value_expr)
    env["vars"][var_name] = value
    print(f"Defined {var_name} = {value}")

def evaluate(expr, local_env=None):
    """Evaluate the parsed Lisp-like expression with debugging."""
    if local_env is None:
        local_env = env["vars"]

    print(f"Evaluating: {expr}")
    try:
        if isinstance(expr, list):
            cmd = expr[0]
            args = [evaluate(arg, local_env) for arg in expr[1:]]
            pos_args, kw_args = parse_keywords(args)

            if cmd in env["commands"]:
                result = env["commands"][cmd](pos_args, kw_args)
                print(f"Command {cmd} result: {result}")
                return result
            elif cmd in local_env:
                func = local_env[cmd]
                result = func(*pos_args, **kw_args) if callable(func) else func
                print(f"User-defined function {cmd} result: {result}")
                return result
            else:
                raise ValueError(f"Unknown command: {cmd}")
        elif isinstance(expr, str):
            result = local_env.get(expr, expr)
            print(f"Variable {expr} result: {result}")
            return result
        else:
            print(f"Literal {expr} result: {expr}")
            return expr
    except Exception as e:
        print(f"Error evaluating expression: {expr}")
        raise e

def add_ai_commands():
    """Define specific AI commands"""
    env["commands"].update({
        "conv2d": lambda args, kwargs: nn.Conv2d(
            in_channels=kwargs["input"],
            out_channels=kwargs["output"],
            kernel_size=kwargs["kernel-size"]
        ),
        "relu": lambda args, kwargs: nn.ReLU(),
        "max-pool": lambda args, kwargs: nn.MaxPool2d(kernel_size=kwargs["kernel-size"]),
        "flatten": lambda args, kwargs: nn.Flatten(),
        "linear": lambda args, kwargs: nn.Linear(
            in_features=kwargs["input"],
            out_features=kwargs["output"]
        ),
        "softmax": lambda args, kwargs: nn.Softmax(dim=1),
        "nn": lambda args, kwargs: nn.Sequential(*args),
        "train": train_model,
        "load-dataset": load_dataset,
        "save": save_model,
        "load-model": load_model,
        "evaluate": evaluate_model,
        "predict": predict_model,
        "print": lambda args, kwargs: print(*args),
    })

def train_model(args, kwargs):
    """Train the model using the specified dataset, loss, and optimizer."""
    model = env["vars"].get(kwargs.get("model")) if isinstance(kwargs.get("model"), str) else kwargs.get("model")
    dataset = kwargs.get("dataset")

    if not model or not dataset:
        raise ValueError("`model` and `dataset` are required for training.")

    loader = DataLoader(dataset, batch_size=kwargs.get("batch_size", 32), shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=kwargs["lr"]) if kwargs["optimizer"] == "sgd" else optim.Adam(model.parameters(), lr=kwargs["lr"])
    loss_fn = nn.CrossEntropyLoss() if kwargs["loss"] == "cross-entropy" else None

    for epoch in range(kwargs["epochs"]):
        epoch_loss = 0
        for inputs, labels in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{kwargs['epochs']} - Loss: {epoch_loss:.4f}")

def load_dataset(args, kwargs):
    """Load the dataset specified in the arguments."""
    dataset_name = args[0]
    augment = kwargs.get("augment", False)

    transform_list = [transforms.ToTensor()]
    if augment:
        transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ] + transform_list

    transform = transforms.Compose(transform_list + [
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if dataset_name == "cifar10":
        return torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    raise ValueError(f"Dataset {dataset_name} not supported")

def save_model(args, kwargs):
    """Save the trained model to the specified path."""
    model, path = args
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(args, kwargs):
    """Load a model from a file."""
    path = args[0]
    if "model" not in kwargs:
        raise ValueError("Please specify the model architecture using ':model' keyword.")
    model = kwargs["model"]
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model

def evaluate_model(args, kwargs):
    """Evaluate the model on a dataset."""
    model = env["vars"].get(kwargs.get("model")) if isinstance(kwargs.get("model"), str) else kwargs.get("model")
    dataset = kwargs.get("dataset")
    if not model or not dataset:
        raise ValueError("`model` and `dataset` are required for evaluation.")
    loader = DataLoader(dataset, batch_size=kwargs.get("batch_size", 32), shuffle=False)
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy

def predict_model(args, kwargs):
    """Make predictions with the model."""
    model = env["vars"].get(args[0]) if isinstance(args[0], str) else args[0]
    inputs = kwargs.get("inputs")

    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        print(f"Predictions: {predicted.tolist()}")
        return predicted.tolist()

def initialize_env():
    # Initialize environment
    env["commands"].update({
        "+": lambda args, kwargs: sum(args),
        "-": lambda args, kwargs: args[0] - sum(args[1:]),
        "*": lambda args, kwargs: torch.prod(torch.tensor(args)),
        "=": lambda args, kwargs: args[0] == args[1],
        "define": lambda args, kwargs: debug_define(args),
        "print": lambda args, kwargs: print(*args),
    })
    add_ai_commands()

def parse_lisp(source):
    """Parse Lisp-like syntax into Python nested lists."""
    import re
    source = re.sub(r";[^\n]*", "", source)
    tokens = re.findall(r'\(|\)|"[^"]*"|[^\s()]+', source)

    def parse_tokens(tokens):
        token = tokens.pop(0)
        if token == '(':
            expr = []
            while tokens[0] != ')':
                expr.append(parse_tokens(tokens))
            tokens.pop(0)
            return expr
        elif token == ')':
            raise SyntaxError("Unexpected ')'")
        elif token.startswith('"') and token.endswith('"'):
            return token[1:-1]
        else:
            try:
                return int(token)
            except ValueError:
                try:
                    return float(token)
                except ValueError:
                    return token

    expressions = []
    while tokens:
        expressions.append(parse_tokens(tokens))
    return expressions

def run_file(filename):
    with open(filename, "r") as file:
        source = file.read()
        print("Source code loaded:")
        print(source)
        program = parse_lisp(source)
        print("Parsed program:")
        print(program)
    for expr in program:
        print("Evaluating expression:", expr)
        evaluate(expr)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ai_lang.py <filename>")
        sys.exit(1)

    initialize_env()
    run_file(sys.argv[1])
