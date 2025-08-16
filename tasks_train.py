# tasks_train.py
import os
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
    Long-running training job executed by the RQ worker.
    Updates job.meta so the Streamlit UI can show live progress.
    Returns minimal info (model path + history).
    """
    job = get_current_job()
    job.meta.update(status="starting", progress=0.0)
    job.save_meta()

    trainer = MLTrainer(model_type=model_type, learning_rate=learning_rate, device=device)

    def _cb(epoch, total_epochs):
        # Update progress + last recorded losses (if available)
        p = max(0.0, min(1.0, float(epoch)/float(total_epochs)))
        meta = {"status": "running", "progress": p, "epoch": epoch, "total_epochs": total_epochs}
        try:
            last_train = trainer.history.get("train_loss", [])
            last_val   = trainer.history.get("val_loss", [])
            if last_train:
                meta["train_loss"] = float(last_train[-1])
            if last_val:
                meta["val_loss"]   = float(last_val[-1])
        except Exception:
            pass
        job.meta.update(**meta)
        job.save_meta()

    # Run training
    trainer.train(
        epochs=epochs,
        batch_size=batch_size,
        samples=samples,
        validation_split=validation_split,
        progress_callback=_cb
    )

    # Save model (shared volume recommended: /app/models)
    model_dir = os.getenv("MODEL_DIR", "/app/models")
    os.makedirs(model_dir, exist_ok=True)
    tag = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_path = os.path.join(model_dir, f"model_{tag}.pt")

    try:
        # If your MLTrainer has save()
        trainer.save(model_path)
    except Exception:
        # Fallbacks
        try:
            import torch
            if hasattr(trainer, "model"):
                torch.save(trainer.model.state_dict(), model_path)
            else:
                raise RuntimeError("No .model to torch.save()")
        except Exception:
            import pickle
            model_path = os.path.join(model_dir, f"trainer_{tag}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(trainer, f)

    # Finalize meta and return result
    job.meta.update(status="finished", model_path=model_path, history=trainer.history)
    job.save_meta()
    return {"model_path": model_path, "history": trainer.history}
