# Project: Chatbot and Web Scraper for Endterm Paper

## Overview
This repository contains the code and resources to replicate the preliminary results for the chatbot and web scraper presented in the endterm paper. The project leverages advanced machine learning models and web scraping techniques to address technical queries and extract valuable insights from web pages.

## Repository Structure
```
├── chatbot/          # Code for the chatbot implementation
├── webscraper/       # Code for the web scraper implementation
├── data/             # Dataset folder (if applicable)
├── doc/              # Auto-generated documentation
├── README.md         # This file
└── requirements.txt  # Python dependencies
```

## Prerequisites
### Software Requirements
- Python 3.8+
- A CUDA-capable GPU (for model acceleration)
- [Apache Kafka](https://kafka.apache.org/) (optional, if needed for future expansions)

### Python Dependencies
Install the required packages using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### Additional Requirements
- Hugging Face account with API token ([Register here](https://huggingface.co/join))
- Pre-installed [PyTorch](https://pytorch.org/) with GPU support

### Required Packages:
- transformers
- langchain
- faiss-cpu
- pandas
- confluent-kafka

### GPU Configuration (Optional)
Ensure that the necessary GPU drivers and CUDA toolkit are installed for model acceleration. Refer to [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/).

## Dataset Information
The chatbot utilizes pre-trained models and embeddings from Hugging Face. The web scraper fetches information directly from URLs and does not rely on pre-existing datasets.

If additional datasets are used and are publicly available, include them in the `data/` directory.

## Instructions to Compile and Run

### Chatbot
#### Step 1: Set Up Hugging Face API Token
1. Obtain your Hugging Face API token.
2. Replace the placeholder `hf_auth` in the code with your actual token.

#### Step 2: Run the Chatbot
1. Navigate to the `chatbot/` directory.
2. Execute the `chatbot.py` script:
   ```bash
   python chatbot.py
   ```

The chatbot will load the pre-trained LLaMA model and process user queries interactively.

### Web Scraper
#### Step 1: Configure URLs
Update the `url_list` variable in the `webscraper.py` script with the URLs you want to scrape.

#### Step 2: Run the Web Scraper
1. Navigate to the `webscraper/` directory.
2. Execute the `webscraper.py` script:
   ```bash
   python webscraper.py
   ```

The results will be saved to a file (default: `output.txt`).

## Documentation
Auto-generated documentation is available in the `doc/` folder. To generate the documentation:
1. Install [Doxygen](https://www.doxygen.nl/index.html) or [Javadoc](https://www.oracle.com/java/technologies/javase/javadoc.html).
2. Run the following command in the root directory:
   ```bash
   doxygen Doxyfile
   ```
The generated HTML files will appear in the `doc/` folder.

## Citation
If you use this code for your research or project, please cite the corresponding paper:
```
@article{YourPaper,
  title={A Digital Cybersecurity Advisor for the Power Industry Using
Open-Source LLM},
  author={Rodney},
  year={2024}
}
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
