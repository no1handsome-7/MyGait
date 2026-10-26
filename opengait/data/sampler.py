import math
import random
import torch
import torch.distributed as dist
import torch.utils.data as tordata


class TripletSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size, batch_shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        if len(self.batch_size) != 2:
            raise ValueError(
                "batch_size should be (P x K) not {}".format(batch_size))
        self.batch_shuffle = batch_shuffle

        self.world_size = dist.get_world_size()
        if (self.batch_size[0]*self.batch_size[1]) % self.world_size != 0:
            raise ValueError("World size ({}) is not divisible by batch_size ({} x {})".format(
                self.world_size, batch_size[0], batch_size[1]))
        self.rank = dist.get_rank()

    def __iter__(self):
        while True:
            sample_indices = []
            pid_list = sync_random_sample_list(
                self.dataset.label_set, self.batch_size[0])

            for pid in pid_list:
                indices = self.dataset.indices_dict[pid]
                indices = sync_random_sample_list(
                    indices, k=self.batch_size[1])
                sample_indices += indices

            if self.batch_shuffle:
                sample_indices = sync_random_sample_list(
                    sample_indices, len(sample_indices))

            total_batch_size = self.batch_size[0] * self.batch_size[1]
            total_size = int(math.ceil(total_batch_size /
                                       self.world_size)) * self.world_size
            sample_indices += sample_indices[:(
                total_batch_size - len(sample_indices))]

            sample_indices = sample_indices[self.rank:total_size:self.world_size]
            yield sample_indices

    def __len__(self):
        return len(self.dataset)


def sync_random_sample_list(obj_list, k, common_choice=False):
    if common_choice:
        idx = random.choices(range(len(obj_list)), k=k) 
        idx = torch.tensor(idx)
    if len(obj_list) < k:
        idx = random.choices(range(len(obj_list)), k=k)
        idx = torch.tensor(idx)
    else:
        idx = torch.randperm(len(obj_list))[:k]
    if torch.cuda.is_available():
        idx = idx.cuda()
    torch.distributed.broadcast(idx, src=0)
    idx = idx.tolist()
    return [obj_list[i] for i in idx]


class InferenceSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        self.size = len(dataset)
        indices = list(range(self.size))

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        if batch_size % world_size != 0:
            raise ValueError("World size ({}) is not divisible by batch_size ({})".format(
                world_size, batch_size))

        if batch_size != 1:
            complement_size = math.ceil(self.size / batch_size) * \
                batch_size
            indices += indices[:(complement_size - self.size)]
            self.size = complement_size

        batch_size_per_rank = int(self.batch_size / world_size)
        indx_batch_per_rank = []

        for i in range(int(self.size / batch_size_per_rank)):
            indx_batch_per_rank.append(
                indices[i*batch_size_per_rank:(i+1)*batch_size_per_rank])

        self.idx_batch_this_rank = indx_batch_per_rank[rank::world_size]

    def __iter__(self):
        yield from self.idx_batch_this_rank

    def __len__(self):
        return len(self.dataset)


class CommonSampler(tordata.sampler.Sampler):
    def __init__(self,dataset,batch_size,batch_shuffle):

        self.dataset = dataset
        self.size = len(dataset)
        self.batch_size = batch_size
        if isinstance(self.batch_size,int)==False:
            raise ValueError(
                "batch_size shoude be (B) not {}".format(batch_size))
        self.batch_shuffle = batch_shuffle
        
        self.world_size = dist.get_world_size()
        if self.batch_size % self.world_size !=0:
            raise ValueError("World size ({}) is not divisble by batch_size ({})".format(
                self.world_size, batch_size))
        self.rank = dist.get_rank() 
    
    def __iter__(self):
        while True:
            indices_list = list(range(self.size))
            sample_indices = sync_random_sample_list(
                    indices_list, self.batch_size, common_choice=True)
            total_batch_size =  self.batch_size
            total_size = int(math.ceil(total_batch_size /
                                       self.world_size)) * self.world_size
            sample_indices += sample_indices[:(
                total_batch_size - len(sample_indices))]
            sample_indices = sample_indices[self.rank:total_size:self.world_size]
            yield sample_indices

    def __len__(self):
        return len(self.dataset)

# **************** For GaitSSB ****************
# Fan, et al: Learning Gait Representation from Massive Unlabelled Walking Videos: A Benchmark, T-PAMI2023
import random
class BilateralSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size, batch_shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_shuffle = batch_shuffle

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.dataset_length = len(self.dataset)
        self.total_indices = list(range(self.dataset_length))

    def __iter__(self):
        random.shuffle(self.total_indices)
        count = 0
        batch_size = self.batch_size[0] * self.batch_size[1]
        while True:
            if (count + 1) * batch_size >= self.dataset_length:
                count = 0
                random.shuffle(self.total_indices)

            sampled_indices = self.total_indices[count*batch_size:(count+1)*batch_size]
            sampled_indices = sync_random_sample_list(sampled_indices, len(sampled_indices))

            total_size = int(math.ceil(batch_size / self.world_size)) * self.world_size
            sampled_indices += sampled_indices[:(batch_size - len(sampled_indices))]

            sampled_indices = sampled_indices[self.rank:total_size:self.world_size]
            count += 1

            yield sampled_indices * 2

    def __len__(self):
        return len(self.dataset)

# **************** Day/Night coverage-aware sampler for LongRange/LRGait ****************

def _lrgait_as_lower_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return [str(v).lower() for v in x]
    return [str(x).lower()]


def _lrgait_type_matches(typ, keywords):
    typ = str(typ).lower()
    return any(str(k).lower() in typ for k in keywords)


class DayNightCoverageTripletSampler(tordata.sampler.Sampler):
    """P x K triplet sampler that boosts night samples without starving day-only IDs.

    Why this sampler exists:
        If only a small subset of identities has night samples, a strict
        "choose only identities with both day and night" strategy will repeatedly
        sample those identities and almost ignore the remaining day-only IDs.

    This sampler separates identity coverage from domain balancing:
        1) Select N identities from the night-capable pool.
        2) Select P-N identities from the non-night/day-only pool when possible.
        3) For night-capable identities, draw a configurable number of night/day
           sequences, e.g. 3 night + 1 day.
        4) For day-only identities, draw K day/all sequences.

    Example for batch_size=[8,4]:
        night_pid_num=5, night_samples_per_night_pid=3,
        day_samples_per_night_pid=1
        => 5 night-capable IDs and 3 day-only IDs per batch.
        => about 15 night samples and 17 day samples per batch.
        This is close to day:night = 1:1 while still sampling day-only IDs.

    Args:
        dataset: OpenGait dataset object.
        batch_size: [P, K], same as TripletSampler.
        batch_shuffle: shuffle the final P*K sequence indices.
        day_keywords/night_keywords: substrings used to parse dataset.types_list.
        night_pid_num: number of identities per batch that must have night data.
        night_pid_ratio: used only when night_pid_num is None.
        night_samples_per_night_pid: night sequences drawn for each night-capable ID.
        day_samples_per_night_pid: day sequences drawn for each night-capable ID.
        day_pid_policy:
            "day_only_first": non-night slots prefer identities without night data.
            "all_without_selected": non-night slots use all identities except selected night IDs.
        fallback:
            "same_id_all": fill missing domain samples using any sequence of same ID.
            "repeat_available": fill missing samples from the available requested domain first.
    """

    def __init__(self, dataset, batch_size, batch_shuffle=False,
                 day_keywords=('day',), night_keywords=('night',),
                 night_pid_num=None, night_pid_ratio=0.5,
                 night_samples_per_night_pid=None,
                 day_samples_per_night_pid=None,
                 day_pid_policy='day_only_first',
                 fallback='same_id_all'):
        self.dataset = dataset
        self.batch_size = batch_size
        if len(self.batch_size) != 2:
            raise ValueError(
                "batch_size should be (P x K) not {}".format(batch_size))
        self.batch_shuffle = batch_shuffle

        self.P = int(self.batch_size[0])
        self.K = int(self.batch_size[1])
        self.day_keywords = _lrgait_as_lower_list(day_keywords)
        self.night_keywords = _lrgait_as_lower_list(night_keywords)
        self.night_pid_num = night_pid_num
        self.night_pid_ratio = float(night_pid_ratio)
        self.day_pid_policy = str(day_pid_policy).lower()
        self.fallback = str(fallback).lower()

        self.world_size = dist.get_world_size()
        if (self.P * self.K) % self.world_size != 0:
            raise ValueError("World size ({}) is not divisible by batch_size ({} x {})".format(
                self.world_size, self.P, self.K))
        self.rank = dist.get_rank()

        self.pid_to_day_indices = {}
        self.pid_to_night_indices = {}
        self.pid_to_other_indices = {}
        for pid in self.dataset.label_set:
            day_indices, night_indices, other_indices = [], [], []
            for idx in self.dataset.indices_dict[pid]:
                typ = self.dataset.types_list[idx]
                if _lrgait_type_matches(typ, self.night_keywords):
                    night_indices.append(idx)
                elif _lrgait_type_matches(typ, self.day_keywords):
                    day_indices.append(idx)
                else:
                    other_indices.append(idx)
            self.pid_to_day_indices[pid] = day_indices
            self.pid_to_night_indices[pid] = night_indices
            self.pid_to_other_indices[pid] = other_indices

        self.night_capable_pids = [
            pid for pid in self.dataset.label_set
            if len(self.pid_to_night_indices[pid]) > 0
        ]
        self.day_only_pids = [
            pid for pid in self.dataset.label_set
            if len(self.pid_to_night_indices[pid]) == 0
        ]
        self.all_pids = list(self.dataset.label_set)

        if self.night_pid_num is None:
            self.night_pid_num = int(round(self.P * self.night_pid_ratio))
        self.night_pid_num = max(0, min(self.P, int(self.night_pid_num)))
        if len(self.night_capable_pids) == 0:
            self.night_pid_num = 0
        self.day_pid_num = self.P - self.night_pid_num

        if night_samples_per_night_pid is None and day_samples_per_night_pid is None:
            self.night_k = int(round(self.K * 0.5))
            self.day_k_for_night_pid = self.K - self.night_k
        elif night_samples_per_night_pid is None:
            self.day_k_for_night_pid = int(day_samples_per_night_pid)
            self.night_k = self.K - self.day_k_for_night_pid
        elif day_samples_per_night_pid is None:
            self.night_k = int(night_samples_per_night_pid)
            self.day_k_for_night_pid = self.K - self.night_k
        else:
            self.night_k = int(night_samples_per_night_pid)
            self.day_k_for_night_pid = int(day_samples_per_night_pid)

        self.night_k = max(0, min(self.K, self.night_k))
        self.day_k_for_night_pid = max(0, min(self.K - self.night_k, self.day_k_for_night_pid))
        if self.night_k + self.day_k_for_night_pid < self.K:
            self.day_k_for_night_pid += self.K - self.night_k - self.day_k_for_night_pid

        print(
            "[DayNightCoverageTripletSampler] "
            f"P={self.P}, K={self.K}, night_pid_num={self.night_pid_num}, "
            f"day_pid_num={self.day_pid_num}, "
            f"night_k={self.night_k}, day_k_for_night_pid={self.day_k_for_night_pid}, "
            f"night_capable_pids={len(self.night_capable_pids)}, "
            f"day_only_pids={len(self.day_only_pids)}"
        )

    def _safe_sample(self, pool, k):
        if k <= 0:
            return []
        if pool is None or len(pool) == 0:
            return []
        return sync_random_sample_list(pool, k=k)

    def _sample_same_pid(self, pid, preferred_pool, k):
        if k <= 0:
            return []
        sampled = self._safe_sample(preferred_pool, k)
        missing = k - len(sampled)
        if missing <= 0:
            return sampled

        day_pool = self.pid_to_day_indices[pid]
        night_pool = self.pid_to_night_indices[pid]
        other_pool = self.pid_to_other_indices[pid]
        all_pool = day_pool + night_pool + other_pool

        if self.fallback == 'repeat_available' and len(preferred_pool) > 0:
            fallback_pool = preferred_pool
        else:
            fallback_pool = all_pool

        sampled += self._safe_sample(fallback_pool, missing)
        return sampled[:k]

    def _sample_for_night_pid(self, pid):
        sampled = []
        sampled += self._sample_same_pid(pid, self.pid_to_night_indices[pid], self.night_k)
        sampled += self._sample_same_pid(pid, self.pid_to_day_indices[pid], self.day_k_for_night_pid)

        missing = self.K - len(sampled)
        if missing > 0:
            all_pool = (self.pid_to_day_indices[pid] +
                        self.pid_to_night_indices[pid] +
                        self.pid_to_other_indices[pid])
            sampled += self._safe_sample(all_pool, missing)
        return sampled[:self.K]

    def _sample_for_day_pid(self, pid):
        day_pool = self.pid_to_day_indices[pid]
        all_pool = (self.pid_to_day_indices[pid] +
                    self.pid_to_night_indices[pid] +
                    self.pid_to_other_indices[pid])
        pool = day_pool if len(day_pool) > 0 else all_pool
        return self._sample_same_pid(pid, pool, self.K)

    def _select_day_pids(self, night_pid_list):
        if self.day_pid_num <= 0:
            return []
        selected_night = set(night_pid_list)

        if self.day_pid_policy == 'day_only_first' and len(self.day_only_pids) > 0:
            candidates = [pid for pid in self.day_only_pids if pid not in selected_night]
            if len(candidates) < self.day_pid_num:
                extra = [pid for pid in self.all_pids if pid not in selected_night and pid not in candidates]
                candidates = candidates + extra
        else:
            candidates = [pid for pid in self.all_pids if pid not in selected_night]

        if len(candidates) == 0:
            candidates = self.all_pids
        return sync_random_sample_list(candidates, self.day_pid_num)

    def __iter__(self):
        while True:
            sample_indices = []

            if self.night_pid_num > 0:
                night_pid_list = sync_random_sample_list(
                    self.night_capable_pids, self.night_pid_num)
            else:
                night_pid_list = []

            day_pid_list = self._select_day_pids(night_pid_list)

            for pid in night_pid_list:
                sample_indices += self._sample_for_night_pid(pid)

            for pid in day_pid_list:
                sample_indices += self._sample_for_day_pid(pid)

            if self.batch_shuffle:
                sample_indices = sync_random_sample_list(
                    sample_indices, len(sample_indices))

            total_batch_size = self.P * self.K
            total_size = int(math.ceil(total_batch_size /
                                       self.world_size)) * self.world_size
            sample_indices += sample_indices[:(
                total_batch_size - len(sample_indices))]
            sample_indices = sample_indices[self.rank:total_size:self.world_size]
            yield sample_indices

    def __len__(self):
        return len(self.dataset)

