from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)  # requires install: pip install transformers

# Installing pyTorch is tricky for M1/2 macbooks and for python 3.13 (as of January 2025)
# You need to have an older version of python (I went with 3.8)

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# Function that is explained in Step 1. below
def simple_text_generation(prompt, model, tokenizer, max_length=100):
    # Encoding prompt to get input ids
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    # Generate text
    output = model.generate(input_ids, max_length=max_length)
    # Decode and return the generated text
    return tokenizer.decode(output[0], skip_special_tokens=True)


# Custom dataset class defined in Step 3. below
class TextDataset(Dataset):
    def __init__(self, input_ids, attention_masks):
        self.input_ids = input_ids  # store input ids
        self.attention_masks = attention_masks  # store attention masks
        self.labels = input_ids.clone()  # required by our GPT-2 model

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
        }


# Function that is explained in Step 5. below
def generate_text(prompt, model, tokenizer, max_length=100):
    # Encoding prompt to get input ids
    inputs = tokenizer.encode_plus(prompt, return_tensors="pt")
    # Extract input ids and attention mask
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Generate text using model
    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, max_length=max_length
    )

    # Decode generated text, skipping special tokens, and return it
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Load dataset (scientific research abstracts related to machine learning)
data = [
    "This paper presents a new method for improving the performance of machine learning models by using data augmentation techniques.",
    "We propose a novel approach to natural language processing that leverages the power of transformers and attention mechanisms.",
    "In this study, we investigate the impact of deep learning algorithms on the accuracy of image recognition tasks.",
    "Our research demonstrates the effectiveness of transfer learning in enhancing the capabilities of neural networks.",
    "This work explores the use of reinforcement learning for optimizing decision-making processes in complex environments.",
    "We introduce a framework for unsupervised learning that significantly reduces the need for labeled data.",
    "The results of our experiments show that ensemble methods can substantially boost model performance.",
    "We analyze the scalability of various machine learning algorithms when applied to large datasets.",
    "Our findings suggest that hyperparameter tuning is crucial for achieving optimal results in machine learning applications.",
    "This research highlights the importance of feature engineering in the context of predictive modeling.",
]


if __name__ == "__main__":

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    # ====================================================================
    # Step 1. Demo of simple text generation
    #
    # Code below demonstrates process and prints intermediate outputs
    # the whole thing is then defined as function simple_text_generation()
    # ====================================================================

    print("\n\nDEMO OF SIMPLE TEXT GENERATION\n")

    # Encode prompt to get input ids
    prompt = "Dear boss ..."
    input_ids = tokenizer.encode(
        prompt, return_tensors="pt"
    )  # pt stands for PyTorch tensors
    print(f"Input IDs: {input_ids[0]}")  # [0] because it's a list within a list
    # Note that warnings get displayed too - nevermind them

    # Generate text using model
    outputs = model.generate(input_ids, max_length=100)
    print(f"\nGenerated Text: \n{outputs[0]}")

    # Now decode these generated tokens based on our initial tokenizer
    decoded_outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nDecoded Text: \n{decoded_outputs}")

    # ====================================================================
    # Step 2. Tokenization
    # ====================================================================

    print("\n\nTOKENIZATION\n")

    # All inputs must have the same length, so we need to "pad them"
    # we do it by adding a dummy token at the end of each input (eos_token)
    tokenizer.pad_token = tokenizer.eos_token
    # Tokenize data
    tokenized_data = [
        tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            max_length=50,
        )
        for sentence in data
    ]
    # Preview
    print(f"\nFirst 2 items of tokenized and padded data: \n{tokenized_data[:2]}")
    # Notice repeating number "50256" - this is our padding token and they
    # don't matter, they disappear when we feed the data to the model
    # Also notice 'attention_mask' key - it tells the model which tokens
    # to pay attention to

    # Now we need to isolate input_ids and attention_masks
    input_ids = [item["input_ids"].squeeze() for item in tokenized_data]
    attention_masks = [item["attention_mask"].squeeze() for item in tokenized_data]
    print(f"\nFirst 2 input_ids: \n{input_ids[:2]}")
    print(f"\nFirst 2 attention_masks: \n{attention_masks[:2]}")

    # And now we convert input ids and attention masks to tensors
    # this is necessary for processing tuned model
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    # Pad all sequences to ensure same length
    padded_input_ids = pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.eos_token_id
    )  # use end-of-sequence token ID as padding value
    # Pad all attention masks as well
    padded_attention_masks = pad_sequence(
        attention_masks, batch_first=True, padding_value=0
    )  # 0 is the default value for padding masks

    print(f"\nFirst 2 padded input_ids: \n{padded_input_ids[:2]}")
    print(f"\nFirst 2 padded attention_masks: \n{padded_attention_masks[:2]}")

    # ====================================================================
    # Step 3. CUSTOM DATASET CLASS
    #
    # Class itself is defined at the top, below imports,
    # here we demonstrate how to use it
    # ====================================================================

    print("\n\nCUSTOM DATASET CLASS\n")

    # Apply class defined at the top
    dataset = TextDataset(padded_input_ids, padded_attention_masks)
    print(f"\nFirst 2 item of dataset: \n{dataset[:2]}")
    # Notice labels and that they are the same as input_ids
    # This is required for our fine tuning

    # ====================================================================
    # Step 4. FINE-TUNING GPT2
    # ====================================================================

    print("\n\nFINE-TUNING GPT2\n")

    # Prepare data in batches using a DataLoader (imported from torch.utils.data)
    # Set batch size to 2 and shuffle data for each epoch
    # batch size 2 results in 5 batches (10 items in total)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Set our GPT2 model to training mode
    model.train()

    # Training loop
    for epoch in range(10):  # epoch is a complete pass through the dataset
        print(f"\nEpoch {epoch + 1}...")
        for batch in dataloader:  # remember, we created 5 batches of size 2 each
            # Unpack input and attention mask ids
            input_ids = batch["input_ids"]
            attention_masks = batch["attention_mask"]
            # Reset gradients to zero after each training batch
            optimizer.zero_grad()
            # Forward pass (processing input and attention masks)
            outputs = model(
                input_ids=input_ids, attention_mask=attention_masks, labels=input_ids
            )
            loss = outputs.loss
            # Backward pass (compute gradiens of the loss)
            loss.backward()
            # Update model parameters
            optimizer.step()

        # Print loss after each epoch to monitor progress
        # we would expect loss to decrease over time
        print(f"...loss: {loss.item()}")

    # ====================================================================
    # Step 5. GENERATE TEXT USING FINE-TUNED MODEL
    #
    # Now that we fine-tuned our model, we can generate text using it
    # Function itself is defined at the top, below imports
    # here we demonstrate how to use it
    # ====================================================================

    print("\n\nGENERATE TEXT USING FINE-TUNED MODEL\n")

    # Test function defined at the top
    prompt = "In this research, we "
    text_generated = generate_text(prompt, model, tokenizer, max_length=100)
    print(f"\nGenerated Text: \n{text_generated}")
