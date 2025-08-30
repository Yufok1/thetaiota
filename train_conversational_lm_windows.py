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
from chat_engine import TinyCausalLM
from transformers import GPT2TokenizerFast
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
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                corpus += f.read() + "\n"
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
    
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


def test_generation(model, tokenizer, epoch, device):
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
                # Encode prompt
                ids = tokenizer.encode(prompt, add_special_tokens=True)
                x = torch.tensor([ids], device=device)
                
                # Generate response
                output = model.generate(x, max_new_tokens=450, temperature=0.8, top_k=20)
                generated_ids = output[0].tolist()[len(ids):]
                response = tokenizer.decode(generated_ids).strip()
                
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
    print(f"\n[DATA] Loading Training Data:")
    patterns = ['**/*.md', '**/*.txt', '**/*.py']
    corpus, files = load_corpus(patterns)
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    ids = tokenizer.encode(corpus, add_special_tokens=True)
    
    print(f"   Files: {len(files)}")
    print(f"   Corpus Size: {len(corpus):,} characters")
    print(f"   Tokens: {len(ids):,}")
    print(f"   Vocabulary: {tokenizer.vocab_size}")
    
    # Create model
    print(f"\n[MODEL] Creating Model:")
    model = TinyCausalLM(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        d_ff=args.d_ff,
        max_len=args.context,
        n_layers=args.n_layers
    )
    model.to(args.device)
    
    # Try to load existing weights
    checkpoint_path = os.path.join('checkpoints', 'tiny_lm.pt')
    if os.path.exists(checkpoint_path):
        try:
            print(f"   Loading existing weights from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=args.device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print(f"   Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                print(f"   Previous loss: {checkpoint.get('loss', 'unknown'):.4f}")
            else:
                model.load_state_dict(checkpoint)
                print(f"   Loaded weights successfully")
        except Exception as e:
            print(f"   Could not load existing weights: {e}")
            print(f"   Starting with fresh weights")
    else:
        print(f"   No existing checkpoint found - starting fresh")
    
    params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {params:,} ({params/1e6:.1f}M)")
    
    # Log VRAM after model creation
    print(f"\n[MEMORY] VRAM After Model Creation:")
    log_memory_usage()
    
    # Optimizer with optional AMP
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    
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
                test_generation(model, tokenizer, epoch, args.device)
        
        # Epoch summary
        avg_epoch_loss = epoch_loss / args.steps_per_epoch
        print(f"\n[EPOCH] Epoch {epoch+1} Complete:")
        print(f"   Average Loss: {avg_epoch_loss:.4f}")
        
        # Test generation quality
        if (epoch + 1) % args.test_every == 0 or epoch == args.epochs - 1:
            test_generation(model, tokenizer, epoch, args.device)
        
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
