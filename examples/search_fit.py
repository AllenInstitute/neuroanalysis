import numpy as np
from neuroanalysis.fitting import Gaussian, SearchFit
import pyqtgraph as pg
pg.dbg()

# make some noise with a gaussian bump
model = Gaussian()
x = np.linspace(0, 100, 1000)
y = np.random.normal(size=len(x)) + model.eval(x=x, xoffset=71, yoffset=0, sigma=5, amp=10)

plt = pg.plot()
plt.plot(x, y, pen=0.5)

# If we attempt to fit with xoffset bounded between 0 and 100, it is likely
# the fit will fail:
fit = model.fit(data=y, x=x, params={'xoffset': (50, 0, 100), 'yoffset': 0, 'amp': (0, -50, 50), 'sigma': (5, 1, 20)})
plt.plot(x, fit.best_fit, pen='r')

# Instead, brute-force search for the best fit over multiple ranges for xoffset and amp:
amp = [{'amp': (-10, -50, 0)}, {'amp': (10, 0, 50)}]
xoffset = [{'xoffset':(x+5, x, x+10)} for x in range(0, 100, 10)]

# Total number of fit attempts is len(amp) * len(xoffset) = 20
search = SearchFit(model, [amp, xoffset], params={'sigma': (5, 1, 20), 'yoffset': 0}, x=x, data=y)

# Optionally, let the user know how the fit is progressing:
with pg.ProgressDialog("Fitting...", maximum=len(search)) as dlg:
    for result in search.iter_fit():
        print("Init params this iteration:", result['params'])
        dlg += 1
        if dlg.wasCanceled():
            raise Exception("User canceled fit")
        
plt.plot(x, search.best_result.best_fit, pen='g')
print("Best fit parameters:", search.best_result.best_values)
