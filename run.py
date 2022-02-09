import numpy as np
import matplotlib.pyplot as plt

# Number Unit: ms

FPS = 30
N_BINS = 200
FRAME_TIME_MU = 0.35 # unit ms
FRAME_TIME_SIGMA = 0.1/2.3263  # 99% in +/- 0.1ms

N_FRAME_PER_WAY = 10000
N_WAYS = 80
WAY_TIME_GAP = 1000 / FPS / 80

WAYS_PER_ENCODER = 10
N_ENCODER = int(N_WAYS / WAYS_PER_ENCODER)
FRAME_TIME = 1000 / FPS / WAYS_PER_ENCODER

def gen_submission_N(size):
  # Probability Distribution {1:P=0.5, {2:10}: P=0.5/9, {>10}: P=0}
  raw = np.ceil(np.random.uniform(0, 18, size))
  return (np.clip(raw, 9, 18) - 8).astype(int)


class FrameSubItem:
  way_idx = None
  frame_idx = None
  is_1st_submission = False
  is_last_submission = False
  submit_time = None
  duration = None

def main():

  # Gen frame time sequence
  frame_time_seq = np.random.normal(FRAME_TIME_MU, FRAME_TIME_SIGMA,
      (N_WAYS, N_FRAME_PER_WAY))

  frame_submission_N = gen_submission_N((N_WAYS, N_FRAME_PER_WAY))

  # Plot the distribution

  fig0 = plt.figure()
  axs0 = fig0.add_subplot(1, 2, 1)
  axs0.hist(frame_time_seq.flatten(), bins=N_BINS)
  axs0.set(xlabel='Latency (ms)', ylabel='Histogram', title='Frame Render Time Distribution')
  axs0.tick_params(axis='x', which='minor')

  axs1 = fig0.add_subplot(1, 2, 2)
  axs1.hist(frame_submission_N.flatten(), bins=np.linspace(0.5, 10.5, 11), edgecolor='black')
  axs1.set(xlabel='No. of Submission', ylabel='Histogram', title='No. of Submission Distribution')

  # Gen submission queue
  entire_submission_queue = []
  sch_time_table = np.zeros((N_WAYS, N_FRAME_PER_WAY))

  for i in range(N_WAYS):

    end_time_of_last_submission = 0
    way_sub_queue = [] # submission queue of each way

    for j in range(N_FRAME_PER_WAY):
      scheduled_start_time = i * WAY_TIME_GAP + j * 1000 / FPS
      sch_time_table[i][j] = scheduled_start_time

      item_duration = frame_time_seq[i, j] / frame_submission_N[i, j]

      for k in range(frame_submission_N[i, j]):
        item = FrameSubItem()
        item.way_idx = i
        item.frame_idx = j

        if k == 0:
          item.is_1st_submission = True
          item.submit_time = scheduled_start_time
          end_time_of_last_submission = item.submit_time + item_duration
        else:
          item.is_1st_submission = False
          item.submit_time = end_time_of_last_submission
          end_time_of_last_submission = item.submit_time + item_duration

        if k == frame_submission_N[i, j] - 1:
          item.is_last_submission = True
        else:
          item.is_last_submission = False

        item.duration = item_duration # duration

        way_sub_queue.append(item) # append to way queue

    entire_submission_queue.append(way_sub_queue)

  # Process the submissions and record frame complete time
  current_time = 0
  current_idx = None
  head_item_plane = []
  complete_time_table = np.zeros((N_WAYS, N_FRAME_PER_WAY))

  for i in range(N_WAYS):
    head_item_plane.append(entire_submission_queue[i].pop(0))

  loop_cnt = 0
  last_sub_cnt = 0
  while(len(head_item_plane) > 0):
    #print("Loop count: " + str(loop_cnt))
    #print("Current Plane Length: " + str(len(head_item_plane)))
    loop_cnt += 1

    # Find the next item to be processed
    for idx in range(len(head_item_plane)):
      item = head_item_plane[idx]

      if idx == 0:
        earliest_submit_time = item.submit_time
        current_idx = idx
      elif item.submit_time < earliest_submit_time:
        earliest_submit_time = item.submit_time
        current_idx = idx

    # Process the earliest item, update the queue
    current_item = head_item_plane[current_idx]
    if current_item.is_1st_submission and current_item.submit_time > current_time:
      current_time = current_item.submit_time
    current_time += current_item.duration

    if current_item.is_last_submission:
      complete_time_table[current_item.way_idx][current_item.frame_idx] = current_time
      last_sub_cnt += 1


    # Update head_item_plane
    if len(entire_submission_queue[current_idx]) > 0:
      # Pop a new item
      head_item_plane[current_idx] = entire_submission_queue[current_idx].pop(0)
    else:
      # Delete this way in plane
      del head_item_plane[current_idx]
      del entire_submission_queue[current_idx]

  print("last sub count: " + str(last_sub_cnt))

  diff_table_GPU = complete_time_table - sch_time_table

  # Plot GPU output
  fig1 = plt.figure()
  axs = fig1.add_subplot(1, 1, 1)
  kwargs = dict(alpha=0.7, density=True, bins=N_BINS)
  l1 = axs.hist(frame_time_seq.flatten(), label='Time w/o blocking', **kwargs)
  l2 = axs.hist(diff_table_GPU.flatten(), label='Time w/ blocking', **kwargs)
  axs.tick_params(axis='x', which='minor')
  axs.set(xlabel='Frame Render Time (ms)', ylabel='Histogram', title='Distribution w/ and w/o blocking')
  axs.legend(['Time w/o blocking', 'Time w/ blocking'])

  # Add video encoder latency
  complete_time_codec = np.zeros((N_WAYS, N_FRAME_PER_WAY))
  current_time_array = np.zeros((N_ENCODER))
  for i in range(N_FRAME_PER_WAY):
    for j in range(N_ENCODER):
      for k in range(WAYS_PER_ENCODER):
        way_idx = WAYS_PER_ENCODER * j + k

        if i == 0 and k == 0:
          current_time_array[j] = complete_time_table[way_idx][i] + FRAME_TIME
          complete_time_codec[way_idx][i] = current_time_array[j]
        else:
          if complete_time_table[way_idx][i] < current_time_array[j]:
            current_time_array[j] += FRAME_TIME
            complete_time_codec[way_idx][i] = current_time_array[j]
          else:
            current_time_array[j] = complete_time_table[way_idx][i] + FRAME_TIME
            complete_time_codec[way_idx][i] = current_time_array[j]

  diff_table_codec = complete_time_codec - complete_time_table

  # Interleave codec output
  WAY_GROUP = WAYS_PER_ENCODER
  complete_time_interleave = np.zeros((N_WAYS, N_FRAME_PER_WAY))
  current_time_array = np.zeros(N_ENCODER)

  for i in range(N_FRAME_PER_WAY):
    for j in range(WAY_GROUP):
      for k in range(N_ENCODER): # 一次做8个，做10次
        way_idx = j * N_ENCODER + k

        if i == 0 and j == 0: # only first column and first group (8 ways)
          current_time_array[k] = complete_time_table[way_idx][i] + FRAME_TIME
          complete_time_interleave[way_idx][i] = current_time_array[k]
        else:
          if complete_time_table[way_idx][i] < current_time_array[k]:
            current_time_array[k] += FRAME_TIME
            complete_time_interleave[way_idx][i] = current_time_array[k]
          else:
            current_time_array[k] = complete_time_table[way_idx][i] + FRAME_TIME
            complete_time_interleave[way_idx][i] = current_time_array[k]

  diff_table_interleave = complete_time_interleave - complete_time_table

  fig2 = plt.figure()
  axs0 = fig2.add_subplot(1, 2, 1)
  axs0.hist(diff_table_codec.flatten(), bins=N_BINS)
  axs0.set(xlabel='Latency (ms)', ylabel='Histogram', title='Latency Distribution w/o Interleave')
  axs0.tick_params(axis='x', which='minor')

  axs1 = fig2.add_subplot(1, 2, 2)
  axs1.hist(diff_table_interleave.flatten(), bins=N_BINS)
  axs1.set(xlabel='Latency (ms)', ylabel='Histogram', title='Latency Distribution w/ Interleave')
  axs1.tick_params(axis='x', which='minor')
  plt.show()

if __name__ == '__main__':
  main()
