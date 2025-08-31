#!/usr/bin/env python3
"""
Windows-compatible conversational LM training with rich logging.
No Unicode emojis for better Windows CMD compatibility.
"""

import os
import glob
import torch
import torch.nn as nn
import argparse
import pickle
from chat_engine import TinyCausalLM
# Use NLTK for lightweight tokenization
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
# Memory usage logging
def log_memory_usage(label=""):
    """Log current memory usage."""
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"[{label}] VRAM: {allocated:.2f}GB (Peak: {peak:.2f}GB)")
    except Exception:
        pass


def load_corpus(file_patterns):
    """Load text corpus from files."""
    corpus = ""
    files = []
    for pattern in file_patterns:
        files.extend(glob.glob(pattern))
    
    total_files = len(files)
    print(f"   Loading {total_files} files...")
    for idx, file_path in enumerate(files):
        # Progress bar
        bar_len = 30
        progress = (idx + 1) / total_files
        filled = int(bar_len * progress)
        bar = '#' * filled + '-' * (bar_len - filled)
        print(f"   [{bar}] {idx+1}/{total_files} : {os.path.basename(file_path)}", end='\r')
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                corpus += f.read() + "\n"
        except Exception as e:
            print(f"\nWarning: Could not read {file_path}: {e}")
    print()  # Newline after progress bar
    
    return corpus, files


def make_batches(ids, context=128, batch_size=8):
    """Create training batches with input/target pairs."""
    while True:
        for _ in range(250):  # steps per "epoch"
            start_indices = torch.randint(0, len(ids) - context - 1, (batch_size,))
            x = torch.stack([torch.tensor(ids[i:i+context]) for i in start_indices])
            y = torch.stack([torch.tensor(ids[i+1:i+context+1]) for i in start_indices])
            yield x, y


def log_rich_training_step(epoch, step, total_steps, loss, avg_loss, model, args):
    """Rich logging for conversational LM training."""
    progress = (step + 1) / total_steps
    
    # Visual progress bar
    bar_length = 20
    filled = int(bar_length * progress)
    bar = "#" * filled + "-" * (bar_length - filled)
    
    print(f"\n[CONV-LM] Training Step")
    print(f"   Epoch {epoch+1}/{args.epochs} | Step {step+1}/{total_steps}")
    print(f"   Progress: [{bar}] {progress*100:.1f}%")
    print(f"   Current Loss: {loss:.4f}")
    print(f"   Average Loss: {avg_loss:.4f}")
    
    # Loss trend analysis
    if hasattr(log_rich_training_step, 'last_loss'):
        delta = log_rich_training_step.last_loss - loss
        if delta > 0.01:
            trend = "IMPROVING"
        elif delta < -0.01:
            trend = "WORSENING"  
        else:
            trend = "STABLE"
        print(f"   Trend: {trend} (delta: {delta:+.4f})")
    log_rich_training_step.last_loss = loss
    
    # VRAM usage
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"   VRAM: {allocated:.2f}GB (Peak: {peak:.2f}GB)")
    except Exception:
        pass
    
    # Model size info
    if step == 0 and epoch == 0:
        params = sum(p.numel() for p in model.parameters())
        print(f"   Model: {params:,} parameters ({params/1e6:.1f}M)")
    
    # Convergence indicators
    if avg_loss < 3.0:
        print(f"   Convergence: EXCELLENT - Learning language patterns!")
    elif avg_loss < 4.0:
        print(f"   Convergence: GOOD - Getting coherent...")
    else:
        print(f"   Convergence: LEARNING - Still learning basics...")


def test_generation(model, vocab, word2idx, epoch, device):
    """Test model generation quality during training."""
    print(f"\n[QUALITY TEST] Testing Generation (Epoch {epoch+1}):")
    
    # Large diverse prompt pool for better conversation testing
    all_prompts = [
        # Greetings & Social
        "Hello, how are you?", "What's up?", "How's your day?", "Good morning!", "Hey there!",
        "Nice to meet you", "How are things?", "What's new?", "How's life?", "Good evening!",
        
        # Personal & Emotional
        "What makes you happy?", "How do you feel?", "What's your mood?", "Are you excited?",
        "What worries you?", "What do you love?", "What makes you laugh?", "Are you lonely?",
        "What's your passion?", "How do you relax?", "What inspires you?", "What motivates you?",
        
        # Knowledge & Learning  
        "What is AI?", "Tell me about science", "How do you learn?", "What's interesting?",
        "Explain creativity", "What is art?", "How does memory work?", "What is consciousness?",
        "Tell me about music", "What is philosophy?", "How do computers think?", "What is intelligence?",
        
        # Stories & Imagination
        "Tell me a story", "What if you could fly?", "Imagine the future", "Create something new",
        "What's your dream?", "Tell me about adventure", "What would you invent?", "Describe magic",
        "What's impossible?", "Tell me about space", "What's your fantasy?", "Create a character",
        
        # Relationships & Social
        "What is friendship?", "Tell me about love", "How do you trust?", "What is family?",
        "How do you communicate?", "What is empathy?", "How do you connect?", "What is kindness?",
        "Tell me about relationships", "How do you care?", "What is loyalty?", "How do you support others?",
        
        # Life & Philosophy
        "What is life?", "What matters most?", "How do you grow?", "What is success?",
        "What is happiness?", "How do you change?", "What is wisdom?", "What is truth?",
        "How do you decide?", "What is meaning?", "How do you cope?", "What is hope?",
        
        # Practical & Helpful
        "Can you help me?", "How do I solve this?", "What should I do?", "Give me advice",
        "How do I improve?", "What's the best way?", "Can you explain?", "Help me understand",
        "What's your opinion?", "How would you approach?", "What do you suggest?", "Can you guide me?",
        
        # Creative & Fun
        "What's funny?", "Tell me a joke", "What's cool?", "What's amazing?", "What's weird?",
        "What's beautiful?", "What's your favorite?", "What's exciting?", "What's surprising?",
        "What's mysterious?", "What's wonderful?", "What's magical?", "What's awesome?"
    ]
    
    # Pick 3 random prompts each time for variety
    import random
    test_prompts = random.sample(all_prompts, 3)
    
    model.eval()
    with torch.no_grad():
        for i, prompt in enumerate(test_prompts):
            try:
                # Encode prompt using NLTK
                prompt_tokens = word_tokenize(prompt)
                ids = [word2idx.get(token, 0) for token in prompt_tokens]
                x = torch.tensor([ids], device=device)

                # Generate response
                output = model.generate(x, max_new_tokens=50, temperature=0.8, top_k=20)
                generated_ids = output[0].tolist()[len(ids):]
                # Decode response using vocab
                response = ' '.join([vocab[idx] for idx in generated_ids if idx < len(vocab)]).strip()

                # Quality assessment
                if response and response != "(thinking...)" and len(response) > 5:
                    quality = "GOOD" if any(word in response.lower() for word in ["i", "am", "is", "the", "and"]) else "OK"
                else:
                    quality = "BAD"
                    response = response or "(empty)"

                print(f"   {i+1}. '{prompt}' -> '{response}' [{quality}]")

            except Exception as e:
                print(f"   {i+1}. '{prompt}' -> Error: {e} [ERROR]")
    model.train()


def main():
    parser = argparse.ArgumentParser(description="Windows-Compatible Conversational LM Training")
    parser.add_argument('--epochs', type=int, default=100, help="Training epochs")
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('--batch', type=int, default=4, help="Batch size (small for 3GB)")
    parser.add_argument('--context', type=int, default=512, help="Context length")
    parser.add_argument('--d_model', type=int, default=1024, help="Model dimension")
    parser.add_argument('--d_ff', type=int, default=4096, help="Feed-forward dimension") 
    parser.add_argument('--n_layers', type=int, default=12, help="Number of layers")
    parser.add_argument('--steps_per_epoch', type=int, default=250, help="Steps per epoch")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device")
    parser.add_argument('--use_amp', action='store_true', help="Use Automatic Mixed Precision")
    parser.add_argument('--test_every', type=int, default=10, help="Test generation every N epochs")
    args = parser.parse_args()
    
    print("==> Enhanced Conversational LM Training")
    print("=" * 50)
    
    # Log initial setup
    print(f"Configuration:")
    print(f"   Device: {args.device}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch}")
    print(f"   Context Length: {args.context}")
    print(f"   Architecture: {args.d_model}d x {args.n_layers} layers")
    print(f"   AMP: {'YES' if args.use_amp else 'NO'}")
    
    os.makedirs('checkpoints', exist_ok=True)
    # Load training data


    cache_dir = os.path.join(os.path.dirname(__file__), 'lm_cache')
    os.makedirs(cache_dir, exist_ok=True)
    corpus_cache = os.path.join(cache_dir, 'corpus.pkl')
    tokens_cache = os.path.join(cache_dir, 'tokens.pkl')
    vocab_cache = os.path.join(cache_dir, 'vocab.pkl')
    ids_cache = os.path.join(cache_dir, 'ids.pkl')

    # Try to load cache
    cache_found = all(os.path.exists(p) for p in [corpus_cache, tokens_cache, vocab_cache, ids_cache])
    if cache_found:
        print(f"[CACHE] Loading cached corpus, tokens, vocab, and ids...")
        with open(corpus_cache, 'rb') as f:
            corpus = pickle.load(f)
        with open(tokens_cache, 'rb') as f:
            tokens = pickle.load(f)
        with open(vocab_cache, 'rb') as f:
            vocab = pickle.load(f)
        with open(ids_cache, 'rb') as f:
            ids = pickle.load(f)
    files = []  # Optionally cache file list if needed
    vocab_size = len(vocab)  # Ensure vocab_size is set when loading from cache
        print(f"[CACHE] Loaded. Token count: {len(tokens):,}, Vocab size: {len(vocab):,}")
    else:
        print(f"\n[DATA] [Step 1/7] Loading Training Data Patterns...")
        patterns = ['**/*.md', '**/*.txt', '**/*.py']
        print(f"[DATA] [Step 2/7] Loading Corpus from Files...")
        corpus, files = load_corpus(patterns)
        print(f"[DATA] [Step 3/7] Corpus Loaded. Size: {len(corpus):,} characters from {len(files)} files.")

        print(f"[DATA] [Step 4/7] Tokenizing Corpus with NLTK...")
        chunk_size = max(1000000, len(corpus) // 50)
        total_len = len(corpus)
        tokens = []
        bar_len = 30
        for i in range(0, total_len, chunk_size):
            chunk = corpus[i:i+chunk_size]
            tokens.extend(word_tokenize(chunk))
            progress = min((i + chunk_size) / total_len, 1.0)
            filled = int(bar_len * progress)
            bar = '#' * filled + '-' * (bar_len - filled)
            print(f"   [TOKENIZE] [{bar}] {int(progress*100):3d}%", end='\r')
        print()
        print(f"[DATA] [Step 5/7] Tokenization Complete. Token count: {len(tokens):,}")

        print(f"[DATA] [Step 6/7] Building Vocabulary and Mapping Tokens...")
        vocab = list(sorted(set(tokens)))
        vocab_size = len(vocab)
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        ids = [word2idx.get(token, 0) for token in tokens]
        print(f"[DATA] Vocabulary Size: {vocab_size} (Full vocabulary restored)")
        print(f"[DATA] Token IDs mapped. Total: {len(ids):,}")

        # Save cache
        print(f"[CACHE] Saving corpus, tokens, vocab, and ids to cache...")
        with open(corpus_cache, 'wb') as f:
            pickle.dump(corpus, f)
        with open(tokens_cache, 'wb') as f:
            pickle.dump(tokens, f)
        with open(vocab_cache, 'wb') as f:
            pickle.dump(vocab, f)
        with open(ids_cache, 'wb') as f:
            pickle.dump(ids, f)

    print(f"[MODEL] [Step 1/4] Instantiating Model...")
    model = TinyCausalLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        d_ff=args.d_ff,
        max_len=args.context,
        n_layers=args.n_layers
    )
    print(f"[MODEL] [Step 2/4] Model Created.")
    model.to(args.device)
    print(f"[MODEL] [Step 3/4] Model Moved to Device: {args.device}")

    checkpoint_path = os.path.join('checkpoints', 'tiny_lm.pt')
    if os.path.exists(checkpoint_path):
        try:
            print(f"[MODEL] [Step 4/4] Loading Existing Weights from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=args.device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print(f"[MODEL] Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                print(f"[MODEL] Previous loss: {checkpoint.get('loss', 'unknown'):.4f}")
            else:
                model.load_state_dict(checkpoint)
                print(f"[MODEL] Loaded weights successfully")
        except Exception as e:
            print(f"[MODEL] Could not load existing weights: {e}")
            print(f"[MODEL] Starting with fresh weights")
    else:
        print(f"[MODEL] No existing checkpoint found - starting fresh")

    params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Parameters: {params:,} ({params/1e6:.1f}M)")

    print(f"[MEMORY] VRAM After Model Creation:")
    log_memory_usage()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"[OPTIMIZER] Created AdamW optimizer.")
    loss_fn = nn.CrossEntropyLoss()
    print(f"[LOSS] CrossEntropyLoss function created.")
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    print(f"[AMP] GradScaler initialized (AMP={'ENABLED' if args.use_amp else 'DISABLED'}).")
    
    print(f"\n" + "="*50)
    print(f"[TRAINING] STARTING - Watch Language Emerge!")
    print(f"="*50)
    
    # Training loop
    model.train()
    batches = make_batches(ids, context=args.context, batch_size=args.batch)
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        
        for step in range(args.steps_per_epoch):
            x, y = next(batches)
            x, y = x.to(args.device), y.to(args.device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with optional AMP
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            
            # Rich logging every 25 steps + live text generation
            if (step + 1) % 25 == 0:
                avg_loss = epoch_loss / (step + 1)
                log_rich_training_step(epoch, step, args.steps_per_epoch, loss.item(), avg_loss, model, args)
                
                # Show actual text generation every 25 steps!
                print(f"\n[LIVE GENERATION] What the AI is saying right now:")
                test_generation(model, vocab, word2idx, epoch, args.device)
        
        # Epoch summary
        avg_epoch_loss = epoch_loss / args.steps_per_epoch
        print(f"\n[EPOCH] Epoch {epoch+1} Complete:")
        print(f"   Average Loss: {avg_epoch_loss:.4f}")
        
        # Test generation quality
        if (epoch + 1) % args.test_every == 0 or epoch == args.epochs - 1:
            test_generation(model, vocab, word2idx, epoch, args.device)
        
        # Save intermediate checkpoints
        if (epoch + 1) % 20 == 0 or epoch == args.epochs - 1:
            save_path = os.path.join('checkpoints', 'tiny_lm.pt')
            torch.save({
                'state_dict': model.state_dict(), 
                'config': vars(args),
                'epoch': epoch + 1,
                'loss': avg_epoch_loss
            }, save_path)
            print(f"   [SAVE] Checkpoint saved: {save_path}")
    
    # Final save
    final_path = os.path.join('checkpoints', 'tiny_lm.pt')
    torch.save({
        'state_dict': model.state_dict(),
        'config': vars(args),
        'epoch': args.epochs,
        'final_loss': avg_epoch_loss
    }, final_path)
    
    print(f"\n" + "="*50)
    print(f"[SUCCESS] TRAINING COMPLETE!")
    print(f"="*50)
    print(f"Model saved: {final_path}")
    print(f"Final loss: {avg_epoch_loss:.4f}")
    print(f"Model size: {params/1e6:.1f}M parameters")
    print(f"Ready for silver-tongued conversation!")
    
    # Final VRAM summary
    print(f"\n[MEMORY] Final VRAM Usage:")
    log_memory_usage()


if __name__ == '__main__':
    main()
