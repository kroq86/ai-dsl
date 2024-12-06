
### **AI DSL: A Domain-Specific Language for Simplified AI Development**

---

### **Overview**
Welcome to **AI DSL**, a domain-specific language designed to simplify the definition, training, evaluation, and deployment of machine learning models. Inspired by Lisp-like syntax, AI DSL abstracts complex AI workflows into a declarative and human-readable format, empowering users to focus on the essence of AI tasks without worrying about boilerplate code.

---

### **Features**
- **Simplified Syntax**: Define neural networks and workflows concisely.
- **Declarative Programming**: Specify *what* to do, not *how* to do it.
- **Reusability**: Save, reload, and reuse models effortlessly.
- **Extensibility**: Add new datasets, layers, or tasks with minimal effort.
- **Debugging-Friendly**: Clear step-by-step outputs for easy debugging.

---

### **Sample Code**
Define, train, and evaluate a CNN on the CIFAR-10 dataset:

```lisp
(define cnn-model
  (nn
    (conv2d :input 3 :output 16 :kernel-size 3)
    (relu)
    (max-pool :kernel-size 2)
    (conv2d :input 16 :output 32 :kernel-size 3)
    (relu)
    (max-pool :kernel-size 2)
    (flatten)
    (linear :input 1152 :output 10)
    (softmax)))

(train cnn-model
  :dataset (load-dataset cifar10)
  :epochs 2
  :optimizer sgd
  :lr 0.01
  :loss cross-entropy)

(evaluate cnn-model
  :dataset (load-dataset cifar10-test))

(save cnn-model cnn_model.ai)

(print "Training complete. Model saved.")
```

---

### **Installation**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/ai-dsl.git
   cd ai-dsl
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the DSL Program**:
   ```bash
   python ai_lang.py program.ai 
   ```

---

### **How It Works**
1. **Define Models**:
   Use the `(define ...)` command to declare your neural network architecture.
2. **Train Models**:
   Specify datasets, optimizers, loss functions, and epochs for training.
3. **Evaluate Models**:
   Test trained models and calculate accuracy on new data.
4. **Save and Reload**:
   Save trained models for later use and reload them seamlessly.

---

### **Supported Commands**
| Command           | Description                                         |
|-------------------|-----------------------------------------------------|
| `define`          | Define a variable or model.                        |
| `train`           | Train a model with specified parameters.           |
| `evaluate`        | Evaluate a model on a dataset.                     |
| `save`            | Save a trained model to a file.                    |
| `load-model`      | Load a model from a file.                          |
| `load-dataset`    | Load datasets like CIFAR-10 for training/evaluation.|
| `conv2d`          | Define a 2D convolutional layer.                   |
| `relu`            | Add a ReLU activation function.                    |
| `max-pool`        | Add a max-pooling layer.                           |
| `linear`          | Define a fully connected layer.                    |
| `softmax`         | Add a softmax activation.                          |
| `print`           | Print debug information.                           |

---

### **Extending the DSL**
To add new commands or features:
1. Edit the `add_ai_commands` function in `ai_lang.py`.
2. Add your custom functionality as a Python lambda or function.
3. Update the README to reflect the new command.

---

### **Future Plans**
- Add support for additional datasets (e.g., ImageNet).
- Extend evaluation metrics (e.g., precision, recall, F1-score).
- Include support for distributed training.
- Improve model visualization capabilities.

---

### **License**
This project is licensed under the **GNU General Public License (GPL)**.  
You are free to use, modify, and distribute this project under the terms of the GPL.  
For more details, see the [LICENSE](LICENSE) file.

---

### **Contributing**
We welcome contributions! Please open issues, submit pull requests, or suggest enhancements via the repository's GitHub page.

