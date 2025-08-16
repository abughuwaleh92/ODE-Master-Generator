# tasks_train.py
import os, json, time
from datetime import datetime
from rq import get_current_job

# Your existing trainer
from src.ml.trainer import MLTrainer

def train_model_job(model_type: str = "pattern_learner",
                    epochs: int = 100,
                    batch_size: int = 32,
                    learning_rate: float = 1e-3,
                    samples: int = 1000,
                    validation_split: float = 0.2,
                    device: str = "cpu"):
    """
    Long-running training job. Updates job.meta for live progress.
    Returns a small JSON result (paths + last metrics).
    """
    job = get_current_job()
    job.meta.update(status="starting", progress=0.0)
    job.save_meta()

    # Init trainer
    trainer = MLTrainer(model_type=model_type, learning_rate=learning_rate, device=device)

    # progress callback used by your current Trainer
    def _cb(epoch, total_epochs):
        p = max(0.0, min(1.0, float(epoch) / float(total_epochs)))
        job.meta.update(status="running", epoch=epoch, total_epochs=total_epochs, progress=p)
        # Optionally expose last loss if available
        try:
            last_train = trainer.history.get("train_loss", [])
            last_val   = trainer.history.get("val_loss", [])
            job.meta.update(train_loss=last_train[-1] if last_train else None,
                            val_loss=last_val[-1] if last_val else None)
        except Exception:
            pass
        job.save_meta()

    # Train
    trainer.train(
        epochs=epochs,
        batch_size=batch_size,
        samples=samples,
        validation_split=validation_split,
        progress_callback=_cb,
    )

    # Persist model
    model_dir = os.getenv("MODEL_DIR", "/app/models")
    os.makedirs(model_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{model_type}_{stamp}"

    # Try trainer.save(), fallback to torch state_dict/pickle
    model_path = os.path.join(model_dir, f"model_{tag}.pt")
    try:
        trainer.save(model_path)  # if your trainer has a save() method
    except Exception:
        try:
            import torch
            if hasattr(trainer, "model"):
                torch.save(trainer.model.state_dict(), model_path)
        except Exception:
            model_path = os.path.join(model_dir, f"trainer_{tag}.pkl")
            import pickle
            with open(model_path, "wb") as f:
                pickle.dump(trainer, f)

    # Final meta
    job.meta.update(status="finished", model_path=model_path, history=trainer.history)
    job.save_meta()

    return {
        "model_path": model_path,
        "history": trainer.history,
        "finished_at": datetime.now().isoformat()
    }
