from torch.utils.data import Sampler
import random

class MultiSiteBatchSampler(Sampler):
    def __init__(self, graphs, batch_size=16, sites_per_batch=2):
        self.graphs = graphs
        self.batch_size = batch_size
        self.sites_per_batch = sites_per_batch
        self.samples_per_site = batch_size // sites_per_batch

        # group graph indices by site
        self.site_to_indices = {}
        for idx, g in enumerate(graphs):
            self.site_to_indices.setdefault(g.site, []).append(idx)

        self.sites = list(self.site_to_indices.keys())

    def __iter__(self):
        n_batches = len(self.graphs) // self.batch_size
        for _ in range(n_batches):
            chosen_sites = random.sample(self.sites, min(self.sites_per_batch, len(self.sites)))
            batch_indices = []
            for site in chosen_sites:
                available = self.site_to_indices[site]
                batch_indices.extend(random.sample(available, min(self.samples_per_site, len(available))))
            random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        return len(self.graphs) // self.batch_size