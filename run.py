import numpy as np
import matplotlib.pyplot as plt

# simulation time unit: ms

FPS = 30
N_BINS = 500
FRAME_TIME_MU = 0.35 # unit ms
FRAME_TIME_SIGMA = 0.1/2.3263  # 99% in +/- 0.1ms

N_FRAME_PER_WAY = 100
N_WAYS = 80

def gen_submission_N(size):
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

  """
  fig, axs = plt.subplots(1, 2)
  axs[0].hist(frame_time_seq.flatten(), bins=N_BINS)
  axs[1].hist(frame_submission_N.flatten(), bins=np.linspace(0.5, 10.5, 11), edgecolor='black')
  plt.show()
  """

  # Gen submission queue
  entire_item_queue = []
  sch_time_table = np.zeros((N_WAYS, N_FRAME_PER_WAY))

  for i in range(N_WAYS):

    end_time_of_last_item = 0
    way_sub_queue = [] # submission queue of each way

    for j in range(N_FRAME_PER_WAY):
      scheduled_start_time = i * FRAME_TIME_MU + j * 1000 / FPS
      sch_time_table[i][j] = scheduled_start_time

      item_duration = frame_time_seq[i, j] / frame_submission_N[i, j]

      for k in range(frame_submission_N[i, j]):
        item = FrameSubItem()
        item.way_idx = i
        item.frame_idx = j

        if k == 0:
          item.is_1st_submission = True
          item.submit_time = scheduled_start_time
          end_time_of_last_item = item.submit_time + item_duration
        else:
          item.is_1st_submission = False
          item.submit_time = end_time_of_last_item
          end_time_of_last_item = item.submit_time + item_duration

        if k == frame_submission_N[i, j] - 1:
          item.is_last_submission = True
        else:
          item.is_last_submission = False

        item.duration = item_duration # duration

        way_sub_queue.append(item) # append to way queue

    entire_item_queue.append(way_sub_queue)

  # Process the submissions and record frame complete time
  current_time = 0
  current_idx = None
  head_item_plane = []
  complete_time_table = np.zeros((N_WAYS, N_FRAME_PER_WAY))

  for i in range(N_WAYS):
    head_item_plane.append(entire_item_queue[i].pop(0))

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
    if len(entire_item_queue[current_idx]) > 0:
      # Pop a new item
      head_item_plane[current_idx] = entire_item_queue[current_idx].pop(0)
    else:
      # Delete this way in plane
      del head_item_plane[current_idx]
      del entire_item_queue[current_idx]

  print("last sub count: " + str(last_sub_cnt))

  diff_table = complete_time_table - sch_time_table

  # Plot GPU output
  fig, axs = plt.subplots()
  axs.hist(diff_table.flatten(), bins=N_BINS)
  plt.show()



if __name__ == '__main__':
  main()
