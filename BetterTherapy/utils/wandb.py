import json
import os
from datetime import datetime
from typing import Any

import bittensor as bt
import matplotlib

import wandb

matplotlib.use("Agg")
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class SubnetEvaluationLogger:
    def __init__(self, validator_config, resume_run_id=None):
        """Initialize wandb with ability to resume previous runs"""

        self.validator_uid = validator_config.get("uid")
        self.validator_hotkey = validator_config.get("hotkey")
        self.subnet_id = validator_config.get("netuid", "unknown")
        self.network = validator_config.get("network", "finney")

        plt.ioff()

        try:
            if resume_run_id:
                self.run = wandb.init(
                    project=f"bittensor-bettertherapy-subnet-{self.network}-{self.subnet_id}",
                    id=resume_run_id,
                    resume="allow",
                    tags=[
                        f"validator-{self.validator_uid}",
                        f"hotkey-{self.validator_hotkey[:8]}",
                        "validator",
                        "bettertherapy",
                        f"network-{self.network}",
                    ],
                )
                bt.logging.info(f"✅ Resumed wandb run: {resume_run_id}")

                self._load_previous_state()

            else:
                run_name = f"validator-{self.validator_uid}-{self.validator_hotkey}-{datetime.now().strftime('%Y%m%d')}"
                self.run = wandb.init(
                    project=f"bittensor-bettertherapy-subnet-{self.subnet_id}",
                    name=run_name,
                    group=f"therapy-evaluations-{datetime.now().strftime('%Y%m%d')}",
                    tags=[
                        f"validator-{self.validator_uid}",
                        f"hotkey-{self.validator_hotkey[:8]}",
                        "validator",
                        "bettertherapy",
                        f"network-{self.network}",
                    ],
                    config={
                        "validator_uid": self.validator_uid,
                        "validator_hotkey": self.validator_hotkey,
                        "subnet_id": self.subnet_id,
                        "network": self.network,
                        "evaluation_interval": "5 minutes",
                    },
                )
                bt.logging.info(f"✅ Created new wandb run: {self.run.id}")
                self._initialize_fresh_state()

            self.run_id = self.run.id
            self._save_run_id()
            self._define_custom_charts()

        except Exception as e:
            bt.logging.error(f"❌ Failed to initialize wandb: {e}")
            self.run = None
            return

        self.all_evaluations = []
        self.request_data = defaultdict(list)
        self.miner_performance = defaultdict(
            lambda: {
                "scores": [],
                "response_times": [],
                "quality_scores": [],
                "timestamps": [],
                "successful_responses": 0,
                "failed_responses": 0,
                "hotkey": "",
            }
        )

        self.leaderboard_data = []
        self.request_comparison_data = []

        # Counters
        self.evaluation_count = 0
        self.successful_responses = 0
        self.failed_responses = 0
        self.unique_requests = set()
        self.unique_miners = set()

    def _define_custom_charts(self):
        """Define custom chart configurations for better UX"""

        wandb.define_metric("request_step")
        wandb.define_metric("request_metrics/*", step_metric="request_step")

        wandb.define_metric("miner_step")
        wandb.define_metric("miner_metrics/*", step_metric="miner_step")

    def log_evaluation_round(
        self, prompt: str, request_id: str, miner_responses: list[dict[str, Any]]
    ):
        """Log evaluation round with improved UX"""

        if not self.run:
            bt.logging.warning("Wandb not initialized, skipping logging")
            return

        timestamp = datetime.now()
        self.evaluation_count += 1
        self.unique_requests.add(request_id)

        bt.logging.info(
            f"📊 Logging evaluation round {self.evaluation_count} for request {request_id}"
        )

        request_metrics = {
            "scores": [],
            "response_times": [],
            "quality_scores": [],
            "miner_uids": [],
        }

        for response in miner_responses:
            miner_uid = response["miner_id"]
            self.unique_miners.add(miner_uid)

            self.request_data[request_id].append(
                {
                    "miner_uid": miner_uid,
                    "total_score": response["total_score"],
                    "quality_score": response["quality_score"],
                    "response_time": response["response_time"],
                    "response_time_score": response["response_time_score"],
                    "timestamp": timestamp,
                }
            )

            self.miner_performance[miner_uid]["scores"].append(response["total_score"])
            self.miner_performance[miner_uid]["response_times"].append(
                response["response_time"]
            )
            self.miner_performance[miner_uid]["quality_scores"].append(
                response["quality_score"]
            )
            self.miner_performance[miner_uid]["timestamps"].append(timestamp)
            self.miner_performance[miner_uid]["successful_responses"] += (
                1 if response["total_score"] > 0 else 0
            )
            self.miner_performance[miner_uid]["hotkey"] = response["hotkey"]

            request_metrics["scores"].append(response["total_score"])
            request_metrics["response_times"].append(response["response_time"])
            request_metrics["quality_scores"].append(response["quality_score"])
            request_metrics["miner_uids"].append(miner_uid)

            self.successful_responses += 1

        bt.logging.info(f"Request metrics: {request_metrics}")
        self.run.log({"request_data": self.request_data})
        try:
            self._create_request_visualizations(request_id, request_metrics, prompt)
        except Exception as e:
            bt.logging.error(f"Failed to create request visualizations: {e}")

        self._update_live_metrics(request_id, request_metrics)

        self._update_leaderboard()

        self._log_request_comparison(request_id, timestamp, prompt, request_metrics)

        if self.evaluation_count % 5 == 0:
            try:
                self._create_miner_comparison_charts()
                self._create_performance_heatmap()
            except Exception as e:
                bt.logging.error(f"Failed to create comparison charts: {e}")

    def _create_request_visualizations(
        self, request_id: str, metrics: dict, prompt: str
    ):
        """Create visualizations for a specific request showing all miners"""

        if not metrics["miner_uids"]:
            return

        fig = plt.figure(figsize=(15, 10))

        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        fig.suptitle(f"Request {request_id} - All Miners Comparison", fontsize=16)

        sorted_indices = np.argsort(metrics["scores"])[::-1]
        sorted_uids = [metrics["miner_uids"][i] for i in sorted_indices]
        sorted_scores = [metrics["scores"][i] for i in sorted_indices]
        sorted_response_times = [metrics["response_times"][i] for i in sorted_indices]
        sorted_quality_scores = [metrics["quality_scores"][i] for i in sorted_indices]

        bars1 = ax1.bar(range(len(sorted_uids)), sorted_scores)
        ax1.set_xlabel("Miner UID")
        ax1.set_ylabel("Total Score")
        ax1.set_title("Total Scores by Miner")
        ax1.set_xticks(range(len(sorted_uids)))
        ax1.set_xticklabels([f"UID {uid}" for uid in sorted_uids], rotation=45)

        colors = [
            "green" if s > np.mean(sorted_scores) else "orange" for s in sorted_scores
        ]
        for bar, color in zip(bars1, colors, strict=False):
            bar.set_color(color)

        bars2 = ax2.bar(range(len(sorted_uids)), sorted_response_times)
        ax2.set_xlabel("Miner UID")
        ax2.set_ylabel("Response Time (s)")
        ax2.set_title("Response Times by Miner")
        ax2.set_xticks(range(len(sorted_uids)))
        ax2.set_xticklabels([f"UID {uid}" for uid in sorted_uids], rotation=45)

        rt_colors = [
            "green" if rt < 5 else "orange" if rt < 15 else "red"
            for rt in sorted_response_times
        ]
        for bar, color in zip(bars2, rt_colors, strict=False):
            bar.set_color(color)

        bars3 = ax3.bar(range(len(sorted_uids)), sorted_quality_scores)
        ax3.set_xlabel("Miner UID")
        ax3.set_ylabel("Quality Score")
        ax3.set_title("Quality Scores by Miner")
        ax3.set_xticks(range(len(sorted_uids)))
        ax3.set_xticklabels([f"UID {uid}" for uid in sorted_uids], rotation=45)

        ax4.axis("off")

        stats_text = f"""Request Summary:
        
Prompt: {prompt[:50]}...
Total Miners: {len(metrics["miner_uids"])}

Best Score: {max(metrics["scores"]):.2f} (UID {sorted_uids[0]})
Avg Score: {np.mean(metrics["scores"]):.2f}
Worst Score: {min(metrics["scores"]):.2f}

Fastest Response: {min(metrics["response_times"]):.2f}s
Slowest Response: {max(metrics["response_times"]):.2f}s
Avg Response Time: {np.mean(metrics["response_times"]):.2f}s

Best Quality: {max(metrics["quality_scores"]):.2f}
Avg Quality: {np.mean(metrics["quality_scores"]):.2f}"""

        ax4.text(
            0.05,
            0.95,
            stats_text,
            transform=ax4.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()

        temp_path = f"/tmp/request_{request_id}.png"
        fig.savefig(temp_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        self.run.log(
            {
                f"request_analysis/{request_id}": wandb.Image(temp_path),
                "request_step": len(self.unique_requests),
            }
        )

        try:
            os.remove(temp_path)
        except:
            pass

    def _update_live_metrics(self, request_id: str, metrics: dict):
        """Update live metrics for real-time monitoring"""

        avg_score = np.mean(metrics["scores"]) if metrics["scores"] else 0
        avg_response_time = (
            np.mean(metrics["response_times"]) if metrics["response_times"] else 0
        )
        avg_quality = (
            np.mean(metrics["quality_scores"]) if metrics["quality_scores"] else 0
        )

        self.run.log(
            {
                "live/total_evaluations": self.evaluation_count,
                "live/unique_requests": len(self.unique_requests),
                "live/unique_miners": len(self.unique_miners),
                "live/success_rate": (
                    (
                        self.successful_responses
                        / (self.successful_responses + self.failed_responses)
                        * 100
                    )
                    if (self.successful_responses + self.failed_responses) > 0
                    else 0
                ),
                "live/latest_request/avg_score": avg_score,
                "live/latest_request/avg_response_time": avg_response_time,
                "live/latest_request/avg_quality": avg_quality,
                "live/latest_request/num_miners": len(metrics["miner_uids"]),
                "live/score_distribution/min": (
                    min(metrics["scores"]) if metrics["scores"] else 0
                ),
                "live/score_distribution/max": (
                    max(metrics["scores"]) if metrics["scores"] else 0
                ),
                "live/score_distribution/std": (
                    np.std(metrics["scores"]) if metrics["scores"] else 0
                ),
            }
        )

    def _update_leaderboard(self):
        """Update live leaderboard table"""

        leaderboard_data = []

        for miner_uid, performance in self.miner_performance.items():
            if not performance["scores"]:
                continue

            avg_score = np.mean(performance["scores"])
            avg_quality = np.mean(performance["quality_scores"])
            avg_response_time = np.mean(performance["response_times"])
            total_requests = len(performance["scores"])
            last_seen = performance["timestamps"][-1].strftime("%Y-%m-%d %H:%M:%S")
            hotkey = performance["hotkey"]
            successful_responses = performance["successful_responses"]

            leaderboard_data.append(
                {
                    "miner_uid": miner_uid,
                    "avg_total_score": avg_score,
                    "avg_quality_score": avg_quality,
                    "avg_response_time": avg_response_time,
                    "total_requests": total_requests,
                    "last_seen": last_seen,
                    "hotkey": hotkey,
                    "successful_responses": successful_responses,
                }
            )

        leaderboard_data.sort(key=lambda x: x["avg_total_score"], reverse=True)

        new_leaderboard_table = wandb.Table(
            columns=[
                "rank",
                "miner_uid",
                "hotkey",
                "avg_total_score",
                "avg_quality_score",
                "avg_response_time",
                "total_requests",
                "success_rate",
                "last_seen",
            ]
        )

        for rank, data in enumerate(leaderboard_data[:20], 1):
            new_leaderboard_table.add_data(
                rank,
                data["miner_uid"],
                f"{data['hotkey'][:6]}...",
                round(data["avg_total_score"], 2),
                round(data["avg_quality_score"], 2),
                round(data["avg_response_time"], 2),
                data["total_requests"],
                str(int(data["successful_responses"] / (data["total_requests"]) * 100))
                + "%",
                data["last_seen"],
            )

        # Log the NEW table
        self.run.log({"leaderboard": new_leaderboard_table})

    def _create_miner_comparison_charts(self):
        """Create charts comparing all miners across all requests"""

        if not self.miner_performance:
            return

        # Create performance over time chart
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 1, hspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])

        fig.suptitle("Miner Performance Over Time", fontsize=16)

        # Plot each miner's score trajectory
        for miner_uid, performance in self.miner_performance.items():
            if len(performance["scores"]) > 1:
                ax1.plot(
                    range(len(performance["scores"])),
                    performance["scores"],
                    label=f"UID {miner_uid}",
                    marker="o",
                    alpha=0.7,
                )

        ax1.set_ylabel("Total Score")
        ax1.set_title("Total Scores Progression")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Plot response times
        for miner_uid, performance in self.miner_performance.items():
            if len(performance["response_times"]) > 1:
                ax2.plot(
                    range(len(performance["response_times"])),
                    performance["response_times"],
                    label=f"UID {miner_uid}",
                    marker="s",
                    alpha=0.7,
                )

        ax2.set_xlabel("Evaluation Round")
        ax2.set_ylabel("Response Time (s)")
        ax2.set_title("Response Times Progression")
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save and log
        temp_path = "/tmp/miner_comparison.png"
        fig.savefig(temp_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        self.run.log(
            {
                "miner_comparison/performance_over_time": wandb.Image(temp_path),
                "miner_step": self.evaluation_count,
            }
        )

        try:
            os.remove(temp_path)
        except:
            pass

    def _create_performance_heatmap(self):
        """Create a heatmap showing miner performance across requests"""

        if len(self.request_data) < 2:
            return

        # Prepare data for heatmap
        miners = sorted(list(self.unique_miners))
        requests = sorted(list(self.unique_requests))[-10:]  # Last 10 requests

        # Create score matrix
        score_matrix = np.full((len(miners), len(requests)), np.nan)

        for j, request_id in enumerate(requests):
            for response in self.request_data[request_id]:
                if response["miner_uid"] in miners:
                    i = miners.index(response["miner_uid"])
                    score_matrix[i, j] = response["total_score"]

        # Create heatmap
        fig = plt.figure(figsize=(12, 8))

        # Mask NaN values
        mask = np.isnan(score_matrix)

        sns.heatmap(
            score_matrix,
            mask=mask,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            xticklabels=[r[-8:] for r in requests],  # Show last 8 chars of request ID
            yticklabels=[f"UID {m}" for m in miners],
            cbar_kws={"label": "Total Score"},
        )

        plt.title("Miner Performance Heatmap (Last 10 Requests)")
        plt.xlabel("Request ID")
        plt.ylabel("Miner")
        plt.tight_layout()

        # Save and log
        temp_path = "/tmp/performance_heatmap.png"
        fig.savefig(temp_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        self.run.log({"analysis/performance_heatmap": wandb.Image(temp_path)})

        try:
            os.remove(temp_path)
        except:
            pass

    def _log_request_comparison(
        self, request_id: str, timestamp: datetime, prompt: str, metrics: dict
    ):
        """Log request comparison data"""

        if not metrics["scores"]:
            return

        # Store data for accumulation
        self.request_comparison_data.append(
            {
                "request_id": request_id[-8:],
                "timestamp": timestamp.strftime("%H:%M:%S"),
                "num_miners": len(metrics["miner_uids"]),
                "best_miner": metrics["miner_uids"][np.argmax(metrics["scores"])],
                "best_score": round(max(metrics["scores"]), 2),
                "avg_score": round(np.mean(metrics["scores"]), 2),
                "score_variance": round(np.var(metrics["scores"]), 2),
                "prompt_preview": prompt[:50] + "...",
            }
        )

        # Create NEW table with all accumulated data
        new_comparison_table = wandb.Table(
            columns=[
                "request_id",
                "timestamp",
                "num_miners",
                "best_miner",
                "best_score",
                "avg_score",
                "score_variance",
                "prompt_preview",
            ]
        )

        # Add all accumulated data to the new table
        for data in self.request_comparison_data[-20:]:  # Last 20 requests
            new_comparison_table.add_data(
                data["request_id"],
                data["timestamp"],
                data["num_miners"],
                data["best_miner"],
                data["best_score"],
                data["avg_score"],
                data["score_variance"],
                data["prompt_preview"],
            )

        # Log the NEW table
        self.run.log({"request_comparison": new_comparison_table})

    def log_error(self, request_id: str, error_message: str):
        """Log errors that occur during evaluation"""

        self.failed_responses += 1

        if self.run:
            self.run.log(
                {
                    "errors/count": self.failed_responses,
                    "errors/latest_request_id": request_id,
                    "errors/latest_message": error_message,
                    "errors/error_rate": (
                        (
                            self.failed_responses
                            / (self.successful_responses + self.failed_responses)
                            * 100
                        )
                        if (self.successful_responses + self.failed_responses) > 0
                        else 0
                    ),
                }
            )

    def create_summary_dashboard(self):
        """Create a summary dashboard (can be called periodically)"""

        if not self.run:
            return

        # First check if we have enough data to create a meaningful dashboard
        all_response_times = []
        all_scores = []
        for perf in self.miner_performance.values():
            all_response_times.extend(perf["response_times"])
            all_scores.extend(perf["scores"])

        # If we don't have enough data, skip creating the dashboard
        if not all_scores or not all_response_times:
            bt.logging.warning("Not enough data to create summary dashboard. Skipping.")
            return

        # Create a comprehensive summary figure
        fig = plt.figure(figsize=(20, 12))

        # Define grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Top performers pie chart
        ax1 = fig.add_subplot(gs[0, 0])
        top_miners = sorted(
            self.miner_performance.items(),
            key=lambda x: np.mean(x[1]["scores"]) if x[1]["scores"] else 0,
            reverse=True,
        )[:5]

        if top_miners and any(np.mean(m[1]["scores"]) > 0 for m in top_miners):
            labels = [f"UID {m[0]}" for m in top_miners]
            sizes = [np.mean(m[1]["scores"]) for m in top_miners]
            ax1.pie(sizes, labels=labels, autopct="%1.1f%%")
            ax1.set_title("Top 5 Miners by Avg Score")
        else:
            ax1.text(
                0.5,
                0.5,
                "No miner performance data yet",
                horizontalalignment="center",
                verticalalignment="center",
            )
            ax1.set_title("Top 5 Miners by Avg Score")

        # 2. Response time distribution
        ax2 = fig.add_subplot(gs[0, 1])
        if all_response_times:
            ax2.hist(
                all_response_times,
                bins=min(20, len(set(all_response_times))),
                alpha=0.7,
                color="blue",
            )
            ax2.set_xlabel("Response Time (s)")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Response Time Distribution")
        else:
            ax2.text(
                0.5,
                0.5,
                "No response time data yet",
                horizontalalignment="center",
                verticalalignment="center",
            )
            ax2.set_title("Response Time Distribution")

        # 3. Score distribution
        ax3 = fig.add_subplot(gs[0, 2])
        if all_scores:
            ax3.hist(
                all_scores, bins=min(20, len(set(all_scores))), alpha=0.7, color="green"
            )
            ax3.set_xlabel("Total Score")
            ax3.set_ylabel("Frequency")
            ax3.set_title("Score Distribution")
        else:
            ax3.text(
                0.5,
                0.5,
                "No score data yet",
                horizontalalignment="center",
                verticalalignment="center",
            )
            ax3.set_title("Score Distribution")

        # 4. Evaluation timeline
        ax4 = fig.add_subplot(gs[1, :])
        eval_times = []
        eval_counts = []

        for i, (req_id, responses) in enumerate(self.request_data.items()):
            if responses:
                eval_times.append(i)
                eval_counts.append(len(responses))

        if eval_times:
            ax4.bar(eval_times, eval_counts, alpha=0.7)
            ax4.set_xlabel("Request Number")
            ax4.set_ylabel("Number of Responses")
            ax4.set_title("Responses per Request Over Time")
        else:
            ax4.text(
                0.5,
                0.5,
                "No evaluation timeline data yet",
                horizontalalignment="center",
                verticalalignment="center",
            )
            ax4.set_title("Responses per Request Over Time")

        # 5. Summary statistics (without emojis)
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis("off")

        success_rate = 0
        if (self.successful_responses + self.failed_responses) > 0:
            success_rate = (
                self.successful_responses
                / (self.successful_responses + self.failed_responses)
                * 100
            )

        mean_score = 0
        std_score = 0
        if all_scores:
            mean_score = np.mean(all_scores)
            std_score = np.std(all_scores)

        mean_response_time = 0
        if all_response_times:
            mean_response_time = np.mean(all_response_times)

        summary_text = f"""Validator Summary Dashboard

    Total Evaluations: {self.evaluation_count}
    Unique Requests: {len(self.unique_requests)}
    Unique Miners: {len(self.unique_miners)}

    Success Rate: {success_rate:.1f}%
    Total Successful Responses: {self.successful_responses}
    Total Failed Responses: {self.failed_responses}

    Average Score Across All: {mean_score:.2f} (std={std_score:.2f})
    Average Response Time: {mean_response_time:.2f}s

    Last Update: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"""

        ax5.text(
            0.5,
            0.5,
            summary_text,
            transform=ax5.transAxes,
            fontsize=14,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
        )

        plt.suptitle(f"Validator {self.validator_uid} - Summary Dashboard", fontsize=20)

        # Add a check before saving to ensure we have valid plots
        temp_path = "/tmp/summary_dashboard.png"
        try:
            fig.savefig(temp_path, dpi=150, bbox_inches="tight")
            self.run.log({"summary/dashboard": wandb.Image(temp_path)})

            try:
                os.remove(temp_path)
            except Exception as _e:
                bt.logging.warning(f"Failed to remove temp file: {temp_path}")
        except Exception as save_error:
            bt.logging.error(f"Failed to save dashboard: {save_error}")
            # Create a simple text-only figure as fallback
            plt.close(fig)
            fig = plt.figure(figsize=(10, 6))
            plt.axis("off")
            plt.text(
                0.5,
                0.5,
                "Error creating dashboard - Insufficient data",
                ha="center",
                va="center",
                fontsize=14,
            )
            plt.savefig(temp_path)
            self.run.log({"summary/dashboard": wandb.Image(temp_path)})
        finally:
            plt.close(fig)

    def _save_run_id(self):
        """Save run ID to a file for auto-resume"""

        config_dir = os.path.expanduser("~/.bittensor/wandb")
        os.makedirs(config_dir, exist_ok=True)

        run_file = os.path.join(
            config_dir,
            f"validator-{self.validator_uid}-{self.validator_hotkey}-{datetime.now().strftime('%Y%m%d')}_run.json",
        )

        run_info = {
            "run_id": self.run_id,
            "validator_uid": self.validator_uid,
            "created_at": datetime.now().isoformat(),
            "project": f"bittensor-bettertherapy-subnet-{self.subnet_id}",
        }

        with open(run_file, "w") as f:
            json.dump(run_info, f, indent=2)

    def _load_run_id(self):
        """Load previous run ID if exists"""

        config_dir = os.path.expanduser("~/.bittensor/wandb")
        run_file = os.path.join(config_dir, f"validator_{self.validator_uid}_run.json")

        if os.path.exists(run_file):
            try:
                with open(run_file) as f:
                    run_info = json.load(f)
                return run_info.get("run_id")
            except:
                return None
        return None

    def _load_previous_state(self):
        """Load state from resumed run"""

        # Get previous counters from run config/summary
        if self.run.summary:
            self.evaluation_count = self.run.summary.get("evaluation_count", 0)
            self.successful_responses = self.run.summary.get("successful_responses", 0)
            self.failed_responses = self.run.summary.get("failed_responses", 0)
        else:
            self._initialize_fresh_state()

        # Initialize data structures (these need to be rebuilt)
        self.all_evaluations = []
        self.request_data = defaultdict(list)
        self.miner_performance = defaultdict(
            lambda: {
                "scores": [],
                "response_times": [],
                "quality_scores": [],
                "timestamps": [],
            }
        )
        self.unique_requests = set()
        self.unique_miners = set()
        self.leaderboard_data = []
        self.request_comparison_data = []

        bt.logging.info(
            f"📊 Loaded previous state: {self.evaluation_count} evaluations"
        )

    def _initialize_fresh_state(self):
        """Initialize fresh state for new run"""

        self.all_evaluations = []
        self.request_data = defaultdict(list)
        self.miner_performance = defaultdict(
            lambda: {
                "scores": [],
                "response_times": [],
                "quality_scores": [],
                "timestamps": [],
            }
        )
        self.leaderboard_data = []
        self.request_comparison_data = []
        self.evaluation_count = 0
        self.successful_responses = 0
        self.failed_responses = 0
        self.unique_requests = set()
        self.unique_miners = set()

    def finish(self):
        """Finish the wandb run if needed"""
        if self.run:
            self.run.finish()
