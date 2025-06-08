import os
import json
import numpy as np
from datetime import datetime
from collections import deque

class TrainingLogger:
    def __init__(self, log_dir="logs", window_size=100):
        self.log_dir = log_dir
        self.window_size = window_size
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_{self.session_id}.json")
        self.csv_file = os.path.join(log_dir, f"scores_{self.session_id}.csv")
        self.debug_file = os.path.join(log_dir, f"debug_{self.session_id}.txt")
        
        # Learning diagnostics
        self.scores_window = deque(maxlen=window_size)
        self.original_scores_window = deque(maxlen=window_size)
        self.epsilon_history = deque(maxlen=window_size)
        self.loss_history = deque(maxlen=1000)
        self.action_distribution = {0: 0, 1: 0, 2: 0, 3: 0}
        self.learning_plateaus = []
        self.convergence_warnings = []
        
        # Performance tracking
        self.best_score = float('-inf')
        self.best_original_score = float('-inf')
        self.episodes_since_improvement = 0
        self.landing_streak = 0
        self.max_landing_streak = 0
        
        # Initialize log data
        self.log_data = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "episodes": [],
            "config": {},
            "diagnostics": {
                "plateaus": [],
                "warnings": [],
                "convergence_issues": []
            }
        }
        
        # Initialize files
        with open(self.csv_file, 'w') as f:
            f.write("episode,score,original_score,epsilon,steps,landing_success,hover_penalty,fuel_used,loss,q_variance\n")
            
        with open(self.debug_file, 'w') as f:
            f.write(f"=== Training Debug Log - {self.session_id} ===\n\n")
    
    def log_config(self, config):
        """Log training configuration"""
        self.log_data["config"] = config
        
    def log_episode(self, episode, score, epsilon, steps, extra_data=None):
        """Log episode data with learning diagnostics"""
        episode_data = {
            "episode": int(episode),
            "score": float(score),
            "epsilon": float(epsilon),
            "steps": int(steps),
            "timestamp": datetime.now().isoformat()
        }
        
        if extra_data:
            # Convert numpy types to Python types for JSON serialization
            clean_extra = {}
            for key, value in extra_data.items():
                if isinstance(value, (list, tuple)):
                    clean_extra[key] = [int(x) if hasattr(x, 'item') else x for x in value[:10]]  # Limit action list size
                elif hasattr(value, 'item'):  # numpy types
                    clean_extra[key] = float(value.item()) if 'float' in str(type(value)) else int(value.item())
                else:
                    clean_extra[key] = float(value) if isinstance(value, (int, float)) else value
            episode_data.update(clean_extra)
            
        self.log_data["episodes"].append(episode_data)
        
        # Update tracking windows
        self.scores_window.append(score)
        self.epsilon_history.append(epsilon)
        
        original_score = extra_data.get("original_reward", score) if extra_data else score
        self.original_scores_window.append(original_score)
        
        # Track action distribution
        actions_taken = extra_data.get("actions_taken", []) if extra_data else []
        for action in actions_taken:
            if action in self.action_distribution:
                self.action_distribution[action] += 1
        
        # Performance tracking
        landing_success = extra_data.get("landing_success", False) if extra_data else False
        
        if score > self.best_score:
            self.best_score = score
            self.episodes_since_improvement = 0
            self.debug_log(f"NEW BEST SCORE: {score:.2f} at episode {episode}")
        else:
            self.episodes_since_improvement += 1
            
        if original_score > self.best_original_score:
            self.best_original_score = original_score
            
        # Landing streak tracking
        if landing_success:
            self.landing_streak += 1
            self.max_landing_streak = max(self.max_landing_streak, self.landing_streak)
        else:
            if self.landing_streak > 0:
                self.debug_log(f"Landing streak ended: {self.landing_streak} consecutive landings")
            self.landing_streak = 0
        
        # Detect learning issues
        if episode > 200:  # Wait for initial learning
            self._detect_learning_issues(episode)
        
        # Write to CSV
        loss = extra_data.get("loss", 0) if extra_data else 0
        q_variance = extra_data.get("q_variance", 0) if extra_data else 0
        hover_penalty = extra_data.get("hover_penalty", 0) if extra_data else 0
        fuel_used = extra_data.get("fuel_used", 0) if extra_data else 0
        
        with open(self.csv_file, 'a') as f:
            f.write(f"{episode},{score:.2f},{original_score:.2f},{epsilon:.4f},{steps},{landing_success},{hover_penalty},{fuel_used},{loss:.4f},{q_variance:.4f}\n")
            
    def _detect_learning_issues(self, episode):
        """Detect common learning problems"""
        if len(self.scores_window) < self.window_size:
            return
            
        current_avg = np.mean(list(self.scores_window)[-50:])
        past_avg = np.mean(list(self.scores_window)[-100:-50])
        
        # Plateau detection (only warn every 100 episodes to reduce spam)
        if abs(current_avg - past_avg) < 5 and self.episodes_since_improvement > 200:
            if self.episodes_since_improvement % 100 == 0:  # Only warn every 100 episodes
                issue = f"PLATEAU: No improvement for {self.episodes_since_improvement} episodes"
                self.debug_log(issue)
                print(f"⚠️  {issue}")
        
        # Oscillation detection
        if len(self.scores_window) >= 20:
            recent_scores = list(self.scores_window)[-20:]
            score_variance = np.var(recent_scores)
            if score_variance > 10000:  # High variance
                issue = f"HIGH VARIANCE: Score oscillating wildly (var={score_variance:.0f})"
                self.debug_log(issue)
        
        # Action distribution analysis
        total_actions = sum(self.action_distribution.values())
        if total_actions > 1000:
            action_probs = {k: v/total_actions for k, v in self.action_distribution.items()}
            if action_probs[0] > 0.9:  # Too much "do nothing"
                issue = f"EXPLORATION: Agent doing nothing {action_probs[0]*100:.1f}% of time"
                self.debug_log(issue)
            elif any(prob < 0.05 for prob in action_probs.values()):
                underused = [k for k, prob in action_probs.items() if prob < 0.05]
                issue = f"ACTION BIAS: Rarely using actions {underused}"
                self.debug_log(issue)
        
        # Epsilon decay issues
        if epsilon := self.epsilon_history[-1]:
            if episode > 1000 and epsilon > 0.5:
                issue = f"SLOW DECAY: Epsilon still {epsilon:.3f} at episode {episode}"
                self.debug_log(issue)
                
    def debug_log(self, message):
        """Write to debug file and print important issues"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(self.debug_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
            
    def log_training_step(self, loss, q_values):
        """Log training step data for convergence monitoring"""
        if loss is not None:
            self.loss_history.append(loss)
            
        # Detect training instability
        if len(self.loss_history) > 100:
            recent_loss = np.mean(list(self.loss_history)[-10:])
            past_loss = np.mean(list(self.loss_history)[-100:-10])
            
            if recent_loss > past_loss * 2:  # Loss increasing
                self.debug_log(f"LOSS SPIKE: Recent loss {recent_loss:.4f} vs past {past_loss:.4f}")
    
    def log_milestone(self, episode, message):
        """Log important milestones"""
        self.debug_log(f"MILESTONE: {message}")
        print(f"🎯 [Episode {episode}] {message}")
        
    def get_learning_summary(self):
        """Get current learning status for debugging"""
        if len(self.scores_window) < 10:
            return "Insufficient data for analysis"
            
        recent_avg = np.mean(list(self.scores_window)[-10:])
        window_avg = np.mean(list(self.scores_window))
        improvement = recent_avg - window_avg
        
        total_actions = sum(self.action_distribution.values())
        action_balance = "balanced" if total_actions == 0 else (
            "unbalanced" if max(self.action_distribution.values()) / total_actions > 0.7 else "balanced"
        )
        
        return {
            "recent_performance": f"{recent_avg:.1f}",
            "trend": "improving" if improvement > 5 else "declining" if improvement < -5 else "stable",
            "episodes_since_best": self.episodes_since_improvement,
            "action_distribution": action_balance,
            "landing_streak": self.landing_streak,
            "max_streak": self.max_landing_streak
        }
        
    def save_log(self):
        """Save complete log to JSON"""
        self.log_data["end_time"] = datetime.now().isoformat()
        
        # Get learning summary and convert any numpy types
        summary = self.get_learning_summary()
        if isinstance(summary, dict):
            clean_summary = {}
            for key, value in summary.items():
                if hasattr(value, 'item'):
                    clean_summary[key] = float(value.item()) if 'float' in str(type(value)) else int(value.item())
                else:
                    clean_summary[key] = value
            self.log_data["final_summary"] = clean_summary
        else:
            self.log_data["final_summary"] = str(summary)
            
        # Clean up any remaining numpy types in the log data
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.log_data, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save JSON log: {e}")
            # Save a simplified version
            simplified_log = {
                "session_id": self.session_id,
                "start_time": self.log_data.get("start_time", ""),
                "end_time": self.log_data.get("end_time", ""),
                "episode_count": len(self.log_data.get("episodes", [])),
                "config": self.log_data.get("config", {})
            }
            with open(self.log_file, 'w') as f:
                json.dump(simplified_log, f, indent=2)
            
    def print_summary(self):
        """Print comprehensive training summary with diagnostics"""
        if not self.log_data["episodes"]:
            return
            
        episodes = self.log_data["episodes"]
        scores = [ep["score"] for ep in episodes]
        original_scores = [ep.get("original_reward", ep["score"]) for ep in episodes]
        
        print("\n" + "="*60)
        print("🚀 MOONLANDER TRAINING SUMMARY")
        print("="*60)
        
        # Basic stats
        print(f"📊 Session: {self.session_id}")
        print(f"📈 Episodes: {len(episodes)}")
        print(f"⏱️  Duration: {(datetime.fromisoformat(episodes[-1]['timestamp']) - datetime.fromisoformat(episodes[0]['timestamp'])).total_seconds()/60:.1f} minutes")
        
        # Performance metrics
        recent_avg = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        print(f"🎯 Recent avg (last 100): {recent_avg:.2f}")
        print(f"🏆 Best shaped score: {max(scores):.2f}")
        print(f"🎮 Best original score: {max(original_scores):.2f}")
        print(f"🎲 Final epsilon: {episodes[-1]['epsilon']:.4f}")
        
        # Landing analysis
        successful_landings = len([ep for ep in episodes if ep.get("landing_success", False)])
        recent_landings = len([ep for ep in episodes[-100:] if ep.get("landing_success", False)])
        print(f"🛬 Total landings: {successful_landings}/{len(episodes)} ({successful_landings/len(episodes)*100:.1f}%)")
        print(f"🛬 Recent landings: {recent_landings}/{min(100, len(episodes))} ({recent_landings/min(100, len(episodes))*100:.1f}%)")
        print(f"🔥 Max landing streak: {self.max_landing_streak}")
        
        # Action distribution
        total_actions = sum(self.action_distribution.values())
        if total_actions > 0:
            print(f"\n🎮 Action Distribution:")
            action_names = {0: "Do Nothing", 1: "Fire Left", 2: "Fire Main", 3: "Fire Right"}
            for action, count in self.action_distribution.items():
                print(f"   {action_names[action]}: {count/total_actions*100:.1f}%")
        
        # Learning diagnostics
        if self.convergence_warnings:
            print(f"\n⚠️  Learning Issues Detected:")
            for warning in self.convergence_warnings[-3:]:  # Show last 3
                print(f"   • {warning}")
        
        # Files
        print(f"\n📁 Log Files:")
        print(f"   JSON: {self.log_file}")
        print(f"   CSV:  {self.csv_file}")
        print(f"   Debug: {self.debug_file}")
        
        # Quick diagnosis
        summary = self.get_learning_summary()
        if isinstance(summary, dict):
            print(f"\n🔍 Quick Diagnosis:")
            print(f"   Trend: {summary['trend'].title()}")
            print(f"   Action balance: {summary['action_distribution'].title()}")
            print(f"   Episodes since best: {summary['episodes_since_best']}")
        
        print("="*60)