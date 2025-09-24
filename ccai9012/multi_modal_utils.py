"""
Multi-Modal Utilities Module
============================

This module provides utilities for working with multi-modal AI models, focusing on vision-language tasks.
It includes tools for image classification using CLIP and visual question answering using Qwen2.5-VL.

The module is organized into two main components:
1. CLIPClassifier: A class for zero-shot image classification using OpenAI's CLIP model
2. VisionQAProcessor: A class for visual question answering and image captioning using Qwen's VL model

These utilities simplify working with pre-trained multi-modal models, handling model loading,
inference, batch processing, and result visualization. The module is particularly useful for:
- Automated image labeling and classification
- Extracting semantic information from images using natural language
- Material detection and attribute extraction from architectural/building images
- Batch processing of image datasets with multi-modal models

Usage:
    ### Image classification with CLIP
    classifier = CLIPClassifier(image_dir="path/to/images")
    results = classifier.batch_classify(["urban", "rural", "industrial"])

    ### Visual question answering with Qwen2.5-VL
    vqa = VisionQAProcessor()
    results = vqa.batch_image_qa("path/to/images", "What materials are used in this building?")
"""

import os
import re
from PIL import Image  # Python Imaging Library for image processing
import torch
from tqdm import tqdm  # Progress bar for tracking batch operations
import pandas as pd
import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPProcessor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info  # Helper utility for Qwen model


# === CLIP Image Classification ===
class CLIPClassifier:
    """
    A class for performing zero-shot image classification using OpenAI's CLIP model.

    CLIP (Contrastive Language-Image Pre-training) is a neural network trained on a variety
    of image-text pairs, enabling it to associate images with natural language descriptions.
    This class provides an interface for using CLIP to classify images based on arbitrary
    text categories without requiring specific training for those categories.

    The class supports:
    - Single image classification with confidence scores
    - Batch processing of multiple images
    - Result visualization
    - Saving classification results to CSV

    Attributes:
        model (CLIPModel): The pre-trained CLIP model from HuggingFace.
        processor (CLIPProcessor): The CLIP processor for preparing inputs.
        device (str): The computing device used for inference ('cuda' or 'cpu').
        image_dir (str): Directory containing images to process.
    """

    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None, image_dir="images"):
        """
        Initialize the CLIP classifier with model, processor and device settings.

        This method loads the CLIP model and processor from HuggingFace's model hub,
        sets up the computing device (automatically selecting CUDA if available),
        and configures the image directory for batch operations.

        Args:
            model_name (str): HuggingFace model identifier for CLIP.
                            Defaults to "openai/clip-vit-base-patch32".
            device (str, optional): Computing device for inference ('cuda' or 'cpu').
                                  If None, automatically uses CUDA if available, else CPU.
            image_dir (str, optional): Directory containing images to be processed.
                                     Defaults to "images".
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.image_dir = image_dir

    def classify_image(self, image_path, text_prompts):
        """
        Classify a single image against a set of text categories using CLIP.

        This method loads an image, processes it with the CLIP model, and determines
        which of the provided text categories best matches the image content. It performs
        zero-shot classification by computing similarity scores between the image and text embeddings.

        Args:
            image_path (str): Path to the image file to be classified.
            text_prompts (list): List of text categories/prompts for zero-shot classification.
                               Example: ["urban landscape", "rural countryside", "industrial area"]

        Returns:
            dict: Classification results containing:
                - filename: The base filename of the image
                - label_id: Index of the best matching text prompt
                - label_text: Text of the best matching prompt
                - confidence: Probability score for the best match
                - all_scores: List of probability scores for all prompts
                Returns None if image loading fails.
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Failed to open image: {image_path}, {e}")
            return None

        inputs = self.processor(text=text_prompts, images=image, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # Image-text similarity scores
            probs = logits_per_image.softmax(dim=1)  # Convert to probabilities

        top_idx = probs.argmax().item()
        return {
            "filename": os.path.basename(image_path),
            "label_id": top_idx,
            "label_text": text_prompts[top_idx],
            "confidence": probs[0, top_idx].item(),
            "all_scores": probs.squeeze().tolist()
        }

    def batch_classify(self, text_prompts, save_csv=None):
        """
        Run inference on all images in the image directory.

        Parameters:
            text_prompts (list): List of text categories for classification
            save_csv (str, optional): Path to save results as CSV file

        Returns:
            DataFrame: Results of classification for all images
        """
        image_paths = [os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) if f.endswith(".jpg")]
        results = []

        for img_path in tqdm(image_paths, desc="Running inference"):
            result = self.classify_image(img_path, text_prompts)
            if result is not None:
                results.append(result)

        df = pd.DataFrame(results)

        if save_csv:
            df.to_csv(save_csv, index=False)
            print(f"Inference results saved to {save_csv}")

        return df

    def show_result(self, df, index=0):
        """
        Visualize an image with its predicted label and confidence.

        Parameters:
            df (DataFrame): Results dataframe from batch_classify
            index (int): Index of the image result to display
        """
        row = df.iloc[index]
        img_path = os.path.join(self.image_dir, row['filename'])
        img = Image.open(img_path)

        plt.figure(figsize=(6,6))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Label: {row['label_text']}\nConfidence: {row['confidence']:.2f}")
        plt.show()


# === Vision QA with Qwen2.5-VL ===
class VisionQAProcessor:
    def __init__(self,
                 model_name="Qwen/Qwen2.5-VL-3B-Instruct",
                 cache_dir="../../../models",
                 keywords_list=None,
                 device=None):
        """
        Initialize the VisionQA processor with model, processor, and optional keyword list.

        Parameters:
            model_name (str): HuggingFace model identifier for Qwen2.5-VL
            cache_dir (str): Directory to cache downloaded models
            keywords_list (list, optional): List of keywords to extract from responses
            device (str): Computing device ('cuda' or 'cpu')
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto", cache_dir=cache_dir
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        if keywords_list is None:
            # Default list of material keywords to extract from responses
            self.keywords_list = [
                "glass", "concrete", "brick", "metal", "wood", "stone", "ceramic",
                "steel", "aluminum", "marble", "plaster", "cladding", "tile",
                "granite", "copper", "composite"
            ]
        else:
            self.keywords_list = keywords_list

    def extract_keywords(self, text):
        """
        Extract material keywords from text by case-insensitive matching.

        Parameters:
            text (str): The text to search for keywords

        Returns:
            list: Found keywords in the text
        """
        text_lower = text.lower()
        found = [k for k in self.keywords_list if re.search(rf"\b{k}\b", text_lower)]
        return found

    def generate_caption_for_image(self, image, prompt):
        """
        Generate a caption / answer for a single image given a prompt.

        Parameters:
            image (PIL.Image): The input image to analyze
            prompt (str): Text prompt/question about the image

        Returns:
            str: Generated text response from the model
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)

        # Remove input_ids prefix to get only the generated response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text

    def batch_image_qa(self, image_folder, prompt, save_csv_path="output/results.csv"):
        """
        Run inference on all images in a folder and save results to CSV.

        Parameters:
            image_folder (str): Directory containing images to analyze
            prompt (str): Text prompt/question to ask about each image
            save_csv_path (str): Path to save the results CSV file

        Returns:
            DataFrame: Results including image filenames, answers and extracted keywords
        """
        results = []
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)

        for filename in tqdm(sorted(os.listdir(image_folder))):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                continue

            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path).convert("RGB")

            output_text = self.generate_caption_for_image(image, prompt)
            keywords = self.extract_keywords(output_text)

            results.append({
                "image": filename,
                "answer": output_text,
                "materials": ", ".join(keywords) if keywords else ""
            })

            tqdm.write(f"Processed {filename} | Materials found: {keywords}")

        df = pd.DataFrame(results)
        df.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
        print(f"All results saved to {save_csv_path}")
        return df
