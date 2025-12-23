class ProgressBar():
    def __init__(self, total,
                 custom_str="",
                 highlight_len=3,
                 bar_len=40,
                 call_times_total=None,
                 ):
        self.i = 0
        self.total = total
        self.custom_str = f" {custom_str}" if custom_str else ""
        self.hl_len = highlight_len
        self.bar_len = bar_len
        self.dig_len = len(f"{self.total}")
        self.loop = "\u2591" * (self.bar_len + self.hl_len) + " " * self.hl_len
        self.call_times = 0
        if call_times_total is not None:
            self.call_times_total = call_times_total
            self.call_times_len = len(f"{call_times_total}")
        else:
            self.call_times_total = None
            self.call_times_len = 0

    def reset(self) -> bool:
        self.i = 0
        self.call_times += 1

    def update(self, done) -> bool:
        self.i += 1
        end_str = "<OUT OF RANGE!>\r" if done > self.total else "\r"
        done = self.total if done > self.total else done
            
        frac = done / self.total
        done_len = int(frac * self.bar_len)
        loop_i = self.i % (self.bar_len + 2 * self.hl_len)
        
        if self.call_times_total is not None:
            call_times_str = f"[{self.call_times:>{self.call_times_len}d}/{self.call_times_total}] "
        else:
            call_times_str = ""

        # Sample:
        #  ┌──────────────────────────────────────────────────────────────────────┐
        #  | > Rollout [  1/100] 99 % ██████████░░░   ░░░░   99/1000 steps <OUT OF RANGE!>|
        #  |└   P1    ┘└   P2   ┘└P3 ┘└   P4   ┘└   P5   ┘└      P6       ┘└   end_str   ┘|
        #  └──────────────────────────────────────────────────────────────────────┘
        print((f" >{self.custom_str} "
               f"{call_times_str}"
               f"{(frac * 100):^3.0f}% "
               f"{'\u2588' * done_len}"
               f"{(self.loop[-loop_i:] + self.loop[:-loop_i])[done_len:-(2 * self.hl_len)]} "
               f"{f'{done:>{self.dig_len}d}'}/{f'{self.total:>{self.dig_len}d}'} steps "),
               end=end_str)
        return True

    def clear(self) -> bool:
        print("\033[2K\r", end="")
        return True