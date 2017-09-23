
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients of the last n iterations
        self.recent_fits = []
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = None
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        # the number of *n* fits / iterations to be used for self.bestx and self.best_fit
        self.recent_n = 10

    def detect(self, warped)
        pass
    
    def sanity_check(self, new_fit, new_xfitted):
        # TBD: validate new fit
        return True

    def append(self, allx, ally):
        pass
