# -*- coding: utf-8 -*-
import numpy as np
from mpi4py import MPI

"""some auxiliary functions for communication."""

DTYPE =  np.float64

class DecentralizedAggregation():
    """Aggregate updates in a decentralized manner."""

    def __init__(self, neighbors_info):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.neighbors_info = neighbors_info
        self.neighbor_ranks = [
            neighbor_rank
            for neighbor_rank in neighbors_info.keys()
            if neighbor_rank != self.rank
        ]
        self.world_size = self.comm.Get_size()

    def agg(self, data, op="weighted_avg", tag=0):
        """Aggregate data using `op` operation.
        Args:
            data (:obj:`torch.Tensor`): A Tensor to be aggragated.
        Returns:
            :obj:`torch.Tensor`: An aggregated tensor.
        """
        # Create some tensors to host the values from neighborhood.
        local_data = {i: np.zeros(data.shape, dtype=DTYPE) for i in self.neighbor_ranks}
        local_data[self.rank] = data

        # async send data.
        reqs = []
        for node_rank in self.neighbor_ranks:
            reqs.append(self.comm.Isend(local_data[self.rank], dest=node_rank, tag=tag))
            reqs.append(self.comm.Irecv(local_data[node_rank], source=node_rank, tag=tag))

        # wait until finish.
        self.complete_wait(reqs)

        if op == "weighted_avg":
            # Aggregate local_data
            output = sum(
                [
                    tensor * self.neighbors_info[rank]
                    for rank, tensor in local_data.items()
                ]
            )
        else:
            raise ValueError("unsupported op {}".format(op))
        return output

    def all_gather_vector(self, data):
        size = self.world_size

        # create buffer to recv
        global_data = np.zeros(len(data) * size)

        self.comm.Allgather(data, global_data)
        return global_data  # dbg: ?? shape

    def all_reduce_vector(self, data, op="avg"):
        size = self.world_size

        reduced_data = np.zeros(data.shape)
        if op == "avg":
            self.comm.Allreduce(data, reduced_data, op=MPI.SUM)
            reduced_data /= size
        elif op == "sum":
            self.comm.Allreduce(data, reduced_data, op=MPI.SUM)
        else:
            raise ValueError("unsupported op {}".format(op))

        return reduced_data

    def gather(self, data, root=0):
        gathered_data = self.comm.gather(data, root=root)
        return gathered_data

    def reduce(self, data, op="avg", root=0):
        size = self.world_size

        if op == "avg":
            reduced_data = self.comm.reduce(data, op=MPI.SUM, root=root)
            reduced_data /= size
        elif op == "sum":
            reduced_data = self.comm.reduce(data, op=MPI.SUM, root=root)
        else:
            raise ValueError("unsupported op {}".format(op))

        return reduced_data

    def complete_wait(self, reqs):
        for req in reqs:
            req.wait()