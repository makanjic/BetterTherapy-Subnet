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
        """Log evaluation round with improved UX and robust metric handling"""

        if not self.run:
            bt.logging.warning("Wandb not initialized, skipping logging")
            return

        def _num(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return None

        def _ensure_perf(uid):
            if uid not in self.miner_performance:
                self.miner_performance[uid] = {
                    "scores": [],
                    "response_times": [],
                    "quality_scores": [],
                    "timestamps": [],
                    "successful_responses": 0,
                    "hotkey": None,
                }

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

        self.request_data.setdefault(request_id, [])

        for response in miner_responses:
            miner_uid = response.get("miner_id")
            if miner_uid is None:
                bt.logging.warning(f"Missing miner_id in response: {response}")
                continue

            self.unique_miners.add(miner_uid)
            _ensure_perf(miner_uid)

            total_score = _num(response.get("total_score"))
            quality_score = _num(response.get("quality_score"))
            response_time = _num(response.get("response_time"))
            response_time_score = _num(response.get("response_time_score"))
            hotkey = response.get("hotkey")

            self.request_data[request_id].append(
                {
                    "miner_uid": miner_uid,
                    "total_score": total_score,
                    "quality_score": quality_score,
                    "response_time": response_time,
                    "response_time_score": response_time_score,
                    "timestamp": timestamp,
                }
            )

            if total_score is not None:
                self.miner_performance[miner_uid]["scores"].append(total_score)
            if response_time is not None:
                self.miner_performance[miner_uid]["response_times"].append(
                    response_time
                )
            if quality_score is not None:
                self.miner_performance[miner_uid]["quality_scores"].append(
                    quality_score
                )

            self.miner_performance[miner_uid]["timestamps"].append(timestamp)
            if (total_score is not None) and (total_score > 0):
                self.miner_performance[miner_uid]["successful_responses"] += 1
            self.miner_performance[miner_uid]["hotkey"] = hotkey

            if total_score is not None:
                request_metrics["scores"].append(total_score)
            if response_time is not None:
                request_metrics["response_times"].append(response_time)
            if quality_score is not None:
                request_metrics["quality_scores"].append(quality_score)
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
        """Create visualizations for a specific request showing all miners (robust)"""
        import math

        uids = list(metrics.get("miner_uids", []))
        sc = list(metrics.get("scores", []))
        rt = list(metrics.get("response_times", []))
        qs = list(metrics.get("quality_scores", []))

        rows = []
        max_len = min(len(uids), len(sc), len(rt), len(qs))
        for i in range(max_len):
            uid, s, t, q = uids[i], sc[i], rt[i], qs[i]

            def bad(x):
                return x is None or (isinstance(x, float) and math.isnan(x))

            if not (bad(s) or bad(t) or bad(q)):
                rows.append((uid, float(s), float(t), float(q)))

        if not rows:
            return

        rows.sort(key=lambda r: r[1], reverse=True)
        sorted_uids = [r[0] for r in rows]
        sorted_scores = [r[1] for r in rows]
        sorted_response_times = [r[2] for r in rows]
        sorted_quality_scores = [r[3] for r in rows]

        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        fig.suptitle(f"Request {request_id} - All Miners Comparison", fontsize=16)

        # Bars
        bars1 = ax1.bar(range(len(rows)), sorted_scores)
        ax1.set_xlabel("Miner UID")
        ax1.set_ylabel("Total Score")
        ax1.set_title("Total Scores by Miner")
        ax1.set_xticks(range(len(rows)))
        ax1.set_xticklabels([f"UID {uid}" for uid in sorted_uids], rotation=45)

        # Color bars by above/below mean (guard len==1)
        mean_score = sum(sorted_scores) / len(sorted_scores)
        for bar, s in zip(bars1, sorted_scores):
            bar.set_color("green" if s > mean_score else "orange")

        bars2 = ax2.bar(range(len(rows)), sorted_response_times)
        ax2.set_xlabel("Miner UID")
        ax2.set_ylabel("Response Time (s)")
        ax2.set_title("Response Times by Miner")
        ax2.set_xticks(range(len(rows)))
        ax2.set_xticklabels([f"UID {uid}" for uid in sorted_uids], rotation=45)

        for bar, rt in zip(bars2, sorted_response_times):
            bar.set_color("green" if rt < 5 else ("orange" if rt < 15 else "red"))

        bars3 = ax3.bar(range(len(rows)), sorted_quality_scores)
        ax3.set_xlabel("Miner UID")
        ax3.set_ylabel("Quality Score")
        ax3.set_title("Quality Scores by Miner")
        ax3.set_xticks(range(len(rows)))
        ax3.set_xticklabels([f"UID {uid}" for uid in sorted_uids], rotation=45)

        # Summary
        ax4.axis("off")
        best_idx = 0  # safe: rows is non-empty
        stats_text = (
            "Request Summary:\n\n"
            f"Prompt: { (prompt or '')[:50]}...\n"
            f"Total Miners: {len(rows)}\n\n"
            f"Best Score: {max(sorted_scores):.2f} (UID {sorted_uids[best_idx]})\n"
            f"Avg Score: {sum(sorted_scores)/len(sorted_scores):.2f}\n"
            f"Worst Score: {min(sorted_scores):!s:.2f}\n\n"
            f"Fastest Response: {min(sorted_response_times):.2f}s\n"
            f"Slowest Response: {max(sorted_response_times):!s:.2f}s\n"
            f"Avg Response Time: {sum(sorted_response_times)/len(sorted_response_times):.2f}s\n\n"
            f"Best Quality: {max(sorted_quality_scores):.2f}\n"
            f"Avg Quality: {sum(sorted_quality_scores)/len(sorted_quality_scores):.2f}"
        )
        ax4.text(
            0.05,
            0.95,
            stats_text,
            transform=ax4.transAxes,
            fontsize=11,
            va="top",
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
        valid_times = [t for t in metrics["response_times"] if t is not None]
        avg_response_time = np.mean(valid_times) if valid_times else 0
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
        try:
            # First check if we have enough data to create a meaningful dashboard
            all_response_times = []
            all_scores = []
            for perf in self.miner_performance.values():
                all_response_times.extend(perf["response_times"])
                all_scores.extend(perf["scores"])

            # If we don't have enough data, skip creating the dashboard
            if not all_scores or not all_response_times:
                bt.logging.warning(
                    "Not enough data to create summary dashboard. Skipping."
                )
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
                    all_scores,
                    bins=min(20, len(set(all_scores))),
                    alpha=0.7,
                    color="green",
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

            plt.suptitle(
                f"Validator {self.validator_uid} - Summary Dashboard", fontsize=20
            )

            # Add a check before saving to ensure we have valid plots
            temp_path = "/tmp/summary_dashboard.png"
            fig.savefig(temp_path, dpi=150, bbox_inches="tight")
            self.run.log({"summary/dashboard": wandb.Image(temp_path)})

            try:
                os.remove(temp_path)
            except Exception as _e:
                bt.logging.warning(f"Failed to remove temp file: {temp_path}")
        except Exception as e:
            bt.logging.error(f"Failed to create summary dashboard: {e}")
            # Create a simple text-only figure as fallback
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
            temp_path = "/tmp/summary_dashboard_error.png"
            fig.savefig(temp_path)
            self.run.log({"summary/dashboard": wandb.Image(temp_path)})
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
