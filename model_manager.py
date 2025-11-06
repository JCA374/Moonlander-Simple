"""
Model management for DQN training.

This module handles model evaluation, saving, and loading logic,
including conservative evaluation criteria to prevent regressive saves.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass

from constants import (
    MODEL_BEST_FILENAME,
    MODEL_FINAL_FILENAME,
    MODEL_CHECKPOINT_PREFIX,
    MODEL_IMPROVING_PREFIX,
    MODEL_BACKUP_PREFIX,
    PERFECT_LANDING_RATE,
    HIGH_LANDING_RATE,
    EARLY_TRAINING_EPISODES
)


@dataclass
class ModelEvaluationResult:
    """Result of model evaluation."""
    should_save: bool
    save_type: str  # "best", "improving", or "checkpoint"
    improvement_reason: str
    new_best_score: float
    new_best_rate: float
    reset_counter: bool


class ModelManager:
    """
    Manages model saving, loading, and evaluation.

    Implements conservative model saving logic that prevents regressive saves:
    - Primary criterion: Landing rate improvement
    - Secondary: Score improvement with same landing rate
    - Special cases: Early training, milestone saves, perfect performance
    """

    def __init__(self, models_dir: str = "models"):
        """
        Initialize model manager.

        Args:
            models_dir: Directory for saving models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Tracking best performance
        self.best_eval_score = float('-inf')
        self.best_landing_rate = 0.0
        self.episodes_since_best = 0

    def evaluate_model(
        self,
        episode: int,
        eval_score: float,
        true_landing_rate: float
    ) -> ModelEvaluationResult:
        """
        Evaluate whether model should be saved.

        Implements conservative evaluation logic:
        1. Landing rate improvement (most important)
        2. Score improvement with same landing rate
        3. Early training flexibility
        4. Milestone saves for long plateaus
        5. Perfect performance checkpoint

        Args:
            episode: Current episode number
            eval_score: Evaluation score
            true_landing_rate: True landing success rate

        Returns:
            ModelEvaluationResult with save decision and details
        """
        is_better_model = False
        improvement_reason = ""
        save_type = "best"
        new_best_score = self.best_eval_score
        new_best_rate = self.best_landing_rate
        reset_counter = False

        # === PRIMARY CRITERIA: TRUE IMPROVEMENTS ===

        # 1. Landing rate improvement (most important)
        if true_landing_rate > self.best_landing_rate:
            is_better_model = True
            save_type = "best"
            improvement_reason = (
                f"Landing rate: {self.best_landing_rate*100:.1f}% -> "
                f"{true_landing_rate*100:.1f}%"
            )
            new_best_rate = true_landing_rate
            new_best_score = eval_score
            reset_counter = True

        # 2. Score improvement with same landing rate (secondary)
        elif true_landing_rate == self.best_landing_rate and eval_score > self.best_eval_score + 20:
            is_better_model = True
            save_type = "best"
            improvement_reason = (
                f"Score: {self.best_eval_score:.1f} -> {eval_score:.1f} "
                f"(same landing rate)"
            )
            new_best_score = eval_score
            reset_counter = True

        # === SPECIAL CASES ===

        # 3. Early training flexibility (only first 1000 episodes)
        elif (episode < EARLY_TRAINING_EPISODES and
              true_landing_rate >= self.best_landing_rate * 0.9 and
              eval_score > self.best_eval_score):
            is_better_model = True
            save_type = "improving"
            improvement_reason = (
                f"Early improvement: Score {self.best_eval_score:.1f} -> {eval_score:.1f}"
            )
            # Don't update best metrics for "improving" saves

        # 4. Milestone saves (much more conservative)
        elif (self.episodes_since_best > 2000 and
              true_landing_rate >= HIGH_LANDING_RATE and
              eval_score > 500):
            # Only save if it's been a REALLY long time and performance is genuinely good
            is_better_model = True
            save_type = "improving"
            improvement_reason = (
                f"Milestone save after {self.episodes_since_best} episodes: "
                f"{true_landing_rate*100:.1f}% success, score {eval_score:.1f}"
            )
            # Don't update best metrics for "improving" saves

        # 5. Perfect performance checkpoint (even if score lower)
        elif true_landing_rate == PERFECT_LANDING_RATE and eval_score > 0:
            is_better_model = True
            save_type = "improving" if true_landing_rate <= self.best_landing_rate else "best"
            improvement_reason = (
                f"Perfect landing rate achieved: 100% success, score {eval_score:.1f}"
            )
            # Update best metrics if it's also a score improvement
            if eval_score > self.best_eval_score or true_landing_rate > self.best_landing_rate:
                save_type = "best"
                new_best_score = max(eval_score, self.best_eval_score)
                new_best_rate = true_landing_rate
                reset_counter = True

        return ModelEvaluationResult(
            should_save=is_better_model,
            save_type=save_type,
            improvement_reason=improvement_reason,
            new_best_score=new_best_score,
            new_best_rate=new_best_rate,
            reset_counter=reset_counter
        )

    def save_model(
        self,
        agent,
        result: ModelEvaluationResult,
        episode: int,
        logger=None
    ) -> Optional[str]:
        """
        Save model based on evaluation result.

        Args:
            agent: DQN agent to save
            result: Model evaluation result
            episode: Current episode number
            logger: Optional training logger

        Returns:
            Path to saved model or None if not saved
        """
        if not result.should_save:
            return None

        # Determine filename based on save type
        if result.save_type == "best":
            save_path = self.models_dir / MODEL_BEST_FILENAME

            # Backup existing best model
            if save_path.exists():
                backup_path = self.models_dir / f"{MODEL_BACKUP_PREFIX}best_{episode}.pth"
                shutil.copy(save_path, backup_path)

            # Update best metrics
            self.best_landing_rate = result.new_best_rate
            self.best_eval_score = result.new_best_score
            if result.reset_counter:
                self.episodes_since_best = 0

        else:  # save_type == "improving"
            save_path = self.models_dir / f"{MODEL_IMPROVING_PREFIX}{episode}.pth"
            # Don't update best metrics for "improving" saves

        # Save the model
        agent.save(str(save_path))

        # Log milestone
        if logger:
            logger.log_milestone(
                episode,
                f"NEW {result.save_type.upper()} MODEL! {result.improvement_reason}"
            )

        print(f"🎯 [Episode {episode}] NEW {result.save_type.upper()} MODEL! "
              f"{result.improvement_reason}")
        print(f"💾 Saved to: {save_path}")

        return str(save_path)

    def save_checkpoint(self, agent, episode: int, logger=None) -> str:
        """
        Save a regular checkpoint (not evaluation-based).

        Args:
            agent: DQN agent to save
            episode: Current episode number
            logger: Optional training logger

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.models_dir / f"{MODEL_CHECKPOINT_PREFIX}{episode}.pth"
        agent.save(str(checkpoint_path))

        if logger:
            progress = 0.0  # Would need total episodes to calculate
            logger.log_milestone(
                episode,
                f"Checkpoint saved at episode {episode}"
            )

        return str(checkpoint_path)

    def save_final_model(self, agent, logger=None) -> str:
        """
        Save the final model at end of training.

        Args:
            agent: DQN agent to save
            logger: Optional training logger

        Returns:
            Path to saved model
        """
        final_path = self.models_dir / MODEL_FINAL_FILENAME
        agent.save(str(final_path))

        if logger:
            logger.log_milestone(-1, "Training completed. Saving final model...")

        return str(final_path)

    def load_best_model(self, agent) -> Tuple[bool, Optional[str]]:
        """
        Load the best saved model.

        Args:
            agent: DQN agent to load into

        Returns:
            Tuple of (success, backup_path)
        """
        best_model_path = self.models_dir / MODEL_BEST_FILENAME

        if not best_model_path.exists():
            return False, None

        print(f"🔄 Loading previous model from {best_model_path}")
        agent.load(str(best_model_path))

        # Reset epsilon for continued training
        agent.epsilon = agent.config.epsilon_resume if hasattr(agent.config, 'epsilon_resume') else 0.6
        agent.epsilon_decay = 0.9995

        # Create backup of loaded model
        timestamp = int(time.time())
        backup_path = self.models_dir / f"{MODEL_BACKUP_PREFIX}{timestamp}.pth"
        shutil.copy(best_model_path, backup_path)
        print(f"📁 Backup saved to {backup_path}")

        return True, str(backup_path)

    def ensure_best_model_exists(self, agent) -> None:
        """
        Ensure a best model file exists (save current if not).

        Args:
            agent: DQN agent to save if no best model exists
        """
        best_model_path = self.models_dir / MODEL_BEST_FILENAME

        if not best_model_path.exists():
            agent.save(str(best_model_path))
            print(f"💾 No best model found, saved current model as best")

    def update_episodes_since_best(self, increment: int = 1) -> None:
        """
        Update the counter for episodes since last best model.

        Args:
            increment: Amount to increment by (default 1)
        """
        self.episodes_since_best += increment

    def get_stats(self) -> dict:
        """
        Get current model manager statistics.

        Returns:
            Dictionary with current stats
        """
        return {
            'best_eval_score': self.best_eval_score,
            'best_landing_rate': self.best_landing_rate,
            'episodes_since_best': self.episodes_since_best,
            'models_dir': str(self.models_dir)
        }

    def debug_model_saving(
        self,
        episode: int,
        eval_score: float,
        true_landing_rate: float
    ) -> Tuple[bool, str, str]:
        """
        Debug why models are or aren't being saved.

        Args:
            episode: Current episode number
            eval_score: Evaluation score
            true_landing_rate: True landing success rate

        Returns:
            Tuple of (should_save, save_type, improvement_reason)
        """
        print(f"🔍 Model Saving Debug (Episode {episode}):")
        print(f"   Current: Score={eval_score:.2f}, "
              f"Landing Rate={true_landing_rate*100:.1f}%")
        print(f"   Best:    Score={self.best_eval_score:.2f}, "
              f"Landing Rate={self.best_landing_rate*100:.1f}%")
        print(f"   Episodes since best: {self.episodes_since_best}")

        # Use the same logic as the actual saving function
        result = self.evaluate_model(episode, eval_score, true_landing_rate)

        if result.should_save:
            print(f"   ✅ WILL SAVE ({result.save_type.upper()}): "
                  f"{result.improvement_reason}")
        else:
            print(f"   ❌ NO SAVE: No significant improvement")

        return result.should_save, result.save_type, result.improvement_reason
