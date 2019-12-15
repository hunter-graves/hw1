# hw1
Post Game Analysis - 12/15/2019

The emphasis of this project was to develop an understanding of the efficiency of
aligning memory such that large vectors could be broken down into smaller quantities and processed in parallel, 
using cpu-localized registers to perform multiplication and addition prior to being stored in memory. 

For some perspective, the serial (non parallel) method of performing matrix multiplication necessitates the use of tradtional memory.
We need to load a row and a column, store the results of the multiplication, store the results of the addition, load the resultant element,
and store the resultant element. If we were small enough to walk alongside our data, we would be tired from trecking back and
forth between the CPU and the memory. We would be tired because at the hardware level, this is a very long distance. 
If we had massive amounts of data, we would be completely exhausted from repeatedly making this trip. Moreover, through this journeying,
we'd also be wasting our CPU's time, because the CPU would be waiting for us to make our journey back and forth before giving us more data
to process. Thankfully, we aren't that small.

Speed is the driving force behind the optimization of this problem, and parallel processing allows us to make use of cache variables 
that exist within a more local proximity to the CPU, and allow us to perform our multiplication and addition at the exact same time. 
Before we can put a jetpack on our data, we have to align our memory such that it may be manipulated by the cache variables. 
Once this alignment is achieved, we are able to use our parallel processing to reduce the amount of time taken to perform matrix
multiplcation, which is extremely important for teaching a computer how to identify a cat as a dog, and vice versa.    


For some (much needed, I would imagine) context, the commits to this repository were a result of my
development environment. The Bridges supercomputer was only accessible via SSH, and I was more confident working in my IDE
and pushing to this repo, pulling from within Bridges, and submitting my process to the batch (Bridges requires batched process
submissions in order to ensure that users don't pull down too much CPU usage).


-H
