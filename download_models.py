from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

def download_model(model_name):
    print(f"\n⬇️ Downloading {model_name}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set environment variables to prevent conversion attempts
    os.environ["SAFETENSORS_FAST_GPU"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    try:
        # Download with explicit safetensors preference
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            use_safetensors=True,
            local_files_only=False,
            force_download=True  # Bypass cache issues
        ).to(device)
        
        print(f"✅ Successfully loaded {model_name}")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Device: {device}")
        
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {str(e)}")
        print("Trying fallback method...")
        
        # Fallback to direct PyTorch weights
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                use_safetensors=False,
                from_tf=False,
                force_download=True
            ).to(device)
            print(f"✅ Fallback successful for {model_name}")
        except Exception as fallback_e:
            print(f"❌ Fallback failed: {str(fallback_e)}")

# Download both models with error handling
download_model('distilbert-base-uncased')
download_model('google/mobilebert-uncased')

print("\n🎉 Model download process completed!")