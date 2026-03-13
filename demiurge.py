import os
import json
import uuid
import logging
import math
from typing import List
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Demiurge:
    """
    Demiurge: The Style Architect.
    Interprets user intent, indexes new posts into a local directory, and 
    retrieves the most stylistically relevant examples from ./as-I-have-written
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", storage_dir: str = "./as-I-have-written"):
        self.model_name = model_name
        self.storage_dir = storage_dir
        
        # Initialize OpenAI Client (requires OPENAI_API_KEY environment variable)
        self.client = OpenAI()
        
        # Ensure the storage directory exists
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
            logger.info(f"Created style repository directory at: {self.storage_dir}")

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Calls the OpenAI Chat Completions API.
        """
        logger.info(f"Calling OpenAI ({self.model_name}) for style analysis...")
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error during OpenAI LLM call: {e}")
            return "generic, professional, safe" # Safe fallback

    def _get_embedding(self, text: str) -> List[float]:
        """
        Calls OpenAI's embedding API using text-embedding-3-small.
        """
        logger.info("Generating vector embedding via OpenAI...")
        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return a zero-vector of the correct dimension (1536 for text-embedding-3-small) as a fallback
            return [0.0] * 1536 

    def index_new_post(self, raw_post_text: str) -> str:
        """
        Takes a raw LinkedIn post, extracts its style fingerprint using an LLM,
        and saves it as a JSON file in the ./as-I-have-written directory.
        """
        logger.info("Extracting style fingerprint for new post...")
        
        system_prompt = """
        Analyze the following social media post. Ignore the topic, product, or industry.
        Extract only the stylistic elements: Tone, pacing, formatting, vocabulary type, and emotional resonance.
        Output a comma-separated list of 10-15 descriptive keywords.
        """
        
        style_keywords = self._call_llm(system_prompt, raw_post_text)
        
        # Get the vector embedding for the extracted keywords
        style_vector = self._get_embedding(style_keywords)
        
        post_data = {
            "style_keywords": [k.strip().lower() for k in style_keywords.split(",")],
            "style_vector": style_vector,
            "content": raw_post_text
        }
        
        # Generate a unique filename
        file_id = f"style_{uuid.uuid4().hex[:8]}.json"
        file_path = os.path.join(self.storage_dir, file_id)
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(post_data, f, indent=4)
            
        logger.info(f"Saved new style reference to {file_path}")
        return file_path

    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculates the Cosine Similarity between two vectors.
        """
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)

    def retrieve_style_references(self, user_instruction: str, top_k: int = 2) -> List[str]:
        """
        Extracts style intent from the user's prompt, scans the local directory,
        and returns the content of the best matching posts.
        """
        logger.info("Demiurge is isolating style intent from user instructions...")
        
        system_prompt = """
        You are a search query generator. The user will give you instructions for a social media post.
        Ignore the topic they want to talk about. Focus ONLY on how they want it to sound.
        Translate their request into a comma-separated list of 5-8 stylistic keywords.
        """
        style_intent = self._call_llm(system_prompt, user_instruction)
        
        # Embed the isolated style intent
        intent_vector = self._get_embedding(style_intent)
        
        logger.info("Searching repository for styles matching intent...")
        
        scored_posts = []
        
        # Scan the directory for saved styles
        for filename in os.listdir(self.storage_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(self.storage_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        
                        # Fallback to empty list if an older post doesn't have a vector yet
                        post_vector = data.get("style_vector", [])
                        score = self._calculate_cosine_similarity(intent_vector, post_vector)
                        
                        scored_posts.append((score, data.get("content", "")))
                except Exception as e:
                    logger.error(f"Error reading {filename}: {e}")

        # Sort by highest score first
        scored_posts.sort(key=lambda x: x[0], reverse=True)
        
        # Return the top_k post contents (if any were found)
        best_matches = [post for score, post in scored_posts[:top_k] if score > 0]
        
        if not best_matches:
            logger.warning("No matching styles found in ./as-I-have-written. Consider indexing more posts.")
            
        return best_matches

# Example Usage:
# if __name__ == "__main__":
#     demiurge = Demiurge()
#     # 1. Index a post to build the repository
#     demiurge.index_new_post("Two years ago, I sat in my car and cried after a pitch meeting. The investor had completely torn apart my vision... But looking back, that brutal feedback was exactly what we needed.")
#     
#     # 2. Retrieve it based on a natural language vibe check
#     matches = demiurge.retrieve_style_references("Write this like a vulnerable founder story.")
#     print(matches)