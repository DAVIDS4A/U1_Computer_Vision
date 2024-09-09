<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>U1 Computer Vision Project</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            background-color: #f4f4f4;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        h1, h2, h3 {
            color: #333;
            text-align: center;
        }
        h1 {
            color: #0A74DA;
        }
        .code-block {
            background-color: #eaeaea;
            padding: 10px;
            border-radius: 5px;
            font-family: 'Courier New', Courier, monospace;
        }
        .button {
            display: inline-block;
            background-color: #0A74DA;
            color: white;
            padding: 10px 20px;
            text-align: center;
            border-radius: 5px;
            text-decoration: none;
            margin: 10px 0;
        }
        .button:hover {
            background-color: #065a9e;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>U1 Computer Vision Project</h1>
        <p>This repository contains the code and resources for the U1 Computer Vision project. The goal of this project is to classify various PC parts using deep learning techniques.</p>

        <h2>Project Overview</h2>
        <p>The project involves preprocessing image data, training a Convolutional Neural Network (CNN) model, and using the trained model for inference on new images. The project integrates with ClearML for experiment tracking and management.</p>

        <h2>Prerequisites</h2>
        <ul>
            <li>Python 3.8+</li>
            <li>PyTorch</li>
            <li>torchvision</li>
            <li>ClearML</li>
            <li>Other dependencies listed in <code>requirements.txt</code></li>
        </ul>

        <h2>Installation</h2>
        <div class="code-block">
            <code>
            git clone https://github.com/DAVIDS4A/U1_Computer_Vision.git<br>
            cd U1_Computer_Vision<br>
            pip install -r requirements.txt
            </code>
        </div>

        <h2>Usage</h2>
        <h3>1. Data Preprocessing</h3>
        <div class="code-block">
            <code>python preprocess.py</code>
        </div>

        <h3>2. Model Training</h3>
        <div class="code-block">
            <code>python train.py</code>
        </div>

        <h3>3. Model Inference</h3>
        <div class="code-block">
            <code>python inference.py</code>
        </div>

        <h2>Example Results</h2>
        <p>Include example results, images, or charts that demonstrate the model's performance.</p>

        <h2>Contributing</h2>
        <p>Feel free to fork this repository, create a new branch, and submit a pull request with your changes.</p>

        <h2>License</h2>
        <p>This project is licensed under the MIT License - see the LICENSE file for details.</p>
    </div>
</body>
</html>
