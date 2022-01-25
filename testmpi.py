from mpi4py import MPI
import tensorflow as tf

comm=MPI.COMM_WORLD
myrank=comm.Get_rank()
ranksize=comm.Get_size()
print("hello world from rank {}, of {} ranks".format(myrank,ranksize))
if tf.device("/gpu:0"):
   print("GPU implementet")
