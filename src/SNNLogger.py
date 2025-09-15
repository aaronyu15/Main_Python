# SNNLogger.py
import logging
import numpy as np
import random
import os
import matplotlib.pyplot as plt

class SNNLogger:
    def __init__(self, log_file="SNN.log", log_level=logging.INFO, save_dir="../logs"):
        self.logger = logging.getLogger("SNNLogger")
        self.logger.setLevel(log_level)
        self.save_dir = save_dir

        ch = logging.StreamHandler()
        ch.setLevel(log_level)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Clear existing log file
        if os.path.exists(os.path.join(self.save_dir, log_file)):
            os.remove(os.path.join(self.save_dir, log_file))    

        fh = logging.FileHandler(os.path.join(self.save_dir, log_file))
        fh.setLevel(log_level)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(ch)
            self.logger.addHandler(fh)

        self.tracked = {}       # {layer_name: [neuron_ids]}
        self.records = {}       # {layer_name: {"potentials": [], "spikes": []}}
        self.timestamps = []    # event timestamps


    # ---------------- Setup ----------------

    def setup_monitoring(self, nn, num_per_layer=5):
        self.tracked.clear()
        self.records.clear()

        #for layer in nn.layers:
        #    if isinstance(layer, Neuron_L):
        #        n = len(layer.mem_map)
        #        ids = random.sample(range(n), min(num_per_layer, n))
        #        self.tracked[layer.name] = ids
        #        self.records[layer.name] = {"potentials": [], "spikes": []}
        #        self.logger.info(f"Monitoring {len(ids)} neurons in {layer.name}: {ids}")

    # ---------------- Recording ----------------
    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def record_args(self, args):
        self.logger.info("Run arguments:")
        for k, v in vars(args).items():
            self.logger.info(f"  {k}: {v}")

    #def record_state(self, nn, event_t):
    #    """
    #    Record neuron states after processing an event at timestamp event_t.
    #    """
    #    self.timestamps.append(event_t)
    
    #    for layer in nn.layer_seq:
    #        if isinstance(layer, Neuron_L):
    #            ids = self.tracked.get(layer.layer_str, [])
    #            if not ids:
    #                continue
    #            
    #            pots = []
    #            spks = []
    
    #            for i in ids:
    #                # use the same indexing as spike check
    #                phys_idx = layer.mem_phys[i]
    #                pot_val = layer.mem_map[i, phys_idx]
    #                pots.append(float(pot_val))
    
    #                # a neuron spikes if potential >= threshold
    #                spk_val = 1 if pot_val >= layer.v_th[i] else 0
    #                spks.append(spk_val)
    
    #            self.records[layer.name]["potentials"].append(pots)
    #            self.records[layer.name]["spikes"].append(spks)


    def summarize_epoch(self, epoch):
        for lname, rec in self.records.items():
            pots = np.array(rec["potentials"])
            spks = np.array(rec["spikes"])
            if pots.size > 0:
                avg_v = float(np.mean(pots))
                total_spikes = int(np.sum(spks))
                self.logger.info(f"[Epoch {epoch}] {lname}: Avg V={avg_v:.3f}, Spikes={total_spikes}")

    # ---------------- Save/Plot ----------------
    def save_activity(self, prefix="activity"):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        for lname, rec in self.records.items():
            np.save(os.path.join(self.save_dir, f"{prefix}_{lname}_pots.npy"), np.array(rec["potentials"]))
            np.save(os.path.join(self.save_dir, f"{prefix}_{lname}_spikes.npy"), np.array(rec["spikes"]))
        self.logger.info(f"Saved activity traces to {self.save_dir}/")

    def plot_membrane_traces(self, layer_name, save_path=None):
        if layer_name not in self.records:
            self.logger.warning(f"No records for {layer_name}")
            return

        pots = np.array(self.records[layer_name]["potentials"])  # [timesteps, neurons]
        ids = self.tracked[layer_name]

        plt.figure(figsize=(10, 5))
        for i, nid in enumerate(ids):
            plt.plot(self.timestamps, pots[:, i], label=f"Neuron {nid}")
        plt.xlabel("Event time (from dataset)")
        plt.ylabel("Membrane potential")
        plt.title(f"Membrane potential traces ({layer_name})")
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def plot_raster(self, layer_name, save_path=None):
        if layer_name not in self.records:
            self.logger.warning(f"No records for {layer_name}")
            return

        spks = np.array(self.records[layer_name]["spikes"])  # [timesteps, neurons]
        ids = self.tracked[layer_name]

        plt.figure(figsize=(10, 5))
        for i, nid in enumerate(ids):
            spike_times = [t for t, s in zip(self.timestamps, spks[:, i]) if s > 0]
            plt.scatter(spike_times, [nid] * len(spike_times), s=5, marker="|")
        plt.xlabel("Event time (from dataset)")
        plt.ylabel("Neuron ID")
        plt.title(f"Spike raster plot ({layer_name})")
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
