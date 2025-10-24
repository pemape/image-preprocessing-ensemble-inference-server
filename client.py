"""
Client script for Fundus Image Preprocessing Inference Server
Demonstrates how to interact with the server API endpoints.
"""

import requests
import base64
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
import time


class FundusPreprocessingClient:
    """
    Client for interacting with the Fundus Image Preprocessing Server.
    """
    
    def __init__(self, server_url: str = "http://localhost:8080"):
        """
        Initialize the client.
        
        Args:
            server_url: Base URL of the preprocessing server
        """
        self.server_url = server_url.rstrip('/')
        
    def health_check(self) -> dict:
        """Check if the server is healthy."""
        try:
            response = requests.get(f"{self.server_url}/")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_server_info(self) -> dict:
        """Get server information and capabilities."""
        try:
            response = requests.get(f"{self.server_url}/info")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_config(self) -> dict:
        """Get current server configuration."""
        try:
            response = requests.get(f"{self.server_url}/config")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_statistics(self) -> dict:
        """Get server statistics."""
        try:
            response = requests.get(f"{self.server_url}/stats")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def process_image(self, image_path: str, image_id: str = None, 
                     return_type: str = "base64", include_metadata: bool = True) -> dict:
        """
        Process a single image.
        
        Args:
            image_path: Path to the image file
            image_id: Optional image identifier
            return_type: "base64" or "array"
            include_metadata: Whether to include metadata in response
            
        Returns:
            Server response dictionary
        """
        try:
            with open(image_path, 'rb') as f:
                files = {'image': f}
                data = {}
                
                if image_id:
                    data['image_id'] = image_id
                data['return_type'] = return_type
                data['include_metadata'] = str(include_metadata).lower()
                
                response = requests.post(f"{self.server_url}/process", files=files, data=data)
                return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def process_batch(self, image_paths: list) -> dict:
        """
        Process multiple images in a batch.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            Server response dictionary
        """
        try:
            files = []
            for image_path in image_paths:
                files.append(('images', open(image_path, 'rb')))
            
            response = requests.post(f"{self.server_url}/process_batch", files=files)
            
            # Close file handles
            for _, file_handle in files:
                file_handle.close()
            
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def save_variants_from_response(self, response: dict, output_dir: str = "./output"):
        """
        Save processed image variants from server response to files.
        
        Args:
            response: Server response containing base64 encoded images
            output_dir: Directory to save images
        """
        if not response.get('success', False):
            print(f"Error in response: {response.get('error', 'Unknown error')}")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        image_id = response.get('image_id', 'unknown')
        variants = response.get('processed_variants', {})
        
        for variant_name, variant_b64 in variants.items():
            try:
                # Decode base64 to image
                image_bytes = base64.b64decode(variant_b64)
                image_array = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                # Save image
                filename = f"{image_id}_{variant_name}.png"
                filepath = output_path / filename
                cv2.imwrite(str(filepath), image)
                
                print(f"Saved {variant_name}: {filepath}")
                
            except Exception as e:
                print(f"Error saving {variant_name}: {e}")


def main():
    """Main function for the client script."""
    parser = argparse.ArgumentParser(description='Fundus Preprocessing Client')
    parser.add_argument('--server', '-s', type=str, default='http://localhost:8080',
                       help='Server URL (default: http://localhost:8080)')
    parser.add_argument('--image', '-i', type=str,
                       help='Path to image file to process')
    parser.add_argument('--batch', '-b', nargs='+',
                       help='Paths to multiple image files for batch processing')
    parser.add_argument('--output', '-o', type=str, default='./client_output',
                       help='Output directory for processed images (default: ./client_output)')
    parser.add_argument('--info', action='store_true',
                       help='Get server information')
    parser.add_argument('--config', action='store_true',
                       help='Get server configuration')
    parser.add_argument('--stats', action='store_true',
                       help='Get server statistics')
    parser.add_argument('--health', action='store_true',
                       help='Check server health')
    
    args = parser.parse_args()
    
    # Initialize client
    client = FundusPreprocessingClient(args.server)
    
    # Health check
    if args.health:
        print("Checking server health...")
        health = client.health_check()
        print(json.dumps(health, indent=2))
        return
    
    # Server info
    if args.info:
        print("Getting server information...")
        info = client.get_server_info()
        print(json.dumps(info, indent=2))
        return
    
    # Server config
    if args.config:
        print("Getting server configuration...")
        config = client.get_config()
        print(json.dumps(config, indent=2))
        return
    
    # Server stats
    if args.stats:
        print("Getting server statistics...")
        stats = client.get_statistics()
        print(json.dumps(stats, indent=2))
        return
    
    # Process single image
    if args.image:
        print(f"Processing image: {args.image}")
        start_time = time.time()
        
        response = client.process_image(args.image, include_metadata=True)
        
        processing_time = time.time() - start_time
        print(f"Request completed in {processing_time:.2f} seconds")
        
        if response.get('success', False):
            print(f"Server processing time: {response.get('processing_time', 0):.2f} seconds")
            print(f"Generated {response.get('variants_count', 0)} variants")
            
            # Save variants
            client.save_variants_from_response(response, args.output)
            
            # Print metadata
            if 'metadata' in response:
                print("\nMetadata:")
                print(json.dumps(response['metadata'], indent=2))
        else:
            print(f"Error: {response.get('error', 'Unknown error')}")
        
        return
    
    # Process batch
    if args.batch:
        print(f"Processing batch of {len(args.batch)} images...")
        start_time = time.time()
        
        response = client.process_batch(args.batch)
        
        processing_time = time.time() - start_time
        print(f"Batch request completed in {processing_time:.2f} seconds")
        
        if response.get('success', False):
            print(f"Server processing time: {response.get('processing_time', 0):.2f} seconds")
            print(f"Successful: {response.get('successful_count', 0)}")
            print(f"Failed: {response.get('failed_count', 0)}")
            
            # Save results from batch
            for result in response.get('results', []):
                if result.get('success', False):
                    print(f"\nProcessing {result['filename']}...")
                    client.save_variants_from_response(result, args.output)
                else:
                    print(f"Failed to process {result['filename']}: {result.get('error', 'Unknown error')}")
        else:
            print(f"Batch error: {response.get('error', 'Unknown error')}")
        
        return
    
    # If no specific action, show usage
    print("Fundus Image Preprocessing Client")
    print("Use --help for usage information")
    print("\nQuick examples:")
    print("  Check server health:    python client.py --health")
    print("  Get server info:        python client.py --info")
    print("  Process single image:   python client.py --image path/to/image.jpg")
    print("  Process batch:          python client.py --batch img1.jpg img2.jpg img3.jpg")


if __name__ == '__main__':
    main()