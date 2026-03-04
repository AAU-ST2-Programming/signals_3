import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.widgets import Slider, Button


def main() -> None:
	x = np.linspace(0, 10, 200)

	defaults = {
		"constant": 0.0,
		"noise_std": 0.0,
		"gain_slope": 0.0,
	}

	rng = np.random.default_rng(7)

	fig, ax = plt.subplots(figsize=(11, 6))
	plt.subplots_adjust(left=0.1, right=0.78, bottom=0.33)
	baseline = np.full_like(x, defaults["constant"])
	signal_line, = ax.plot(x, baseline, lw=2, label="Signal")
	baseline_line, = ax.plot(x, np.zeros_like(x), "--", lw=1.2, alpha=0.7, label="Constant baseline")

	ax.set_title("Interactive Constant Signal with Noise and Gain")
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.grid(alpha=0.35)
	ax.legend(loc="upper left")

	ax_constant = plt.axes((0.1, 0.2, 0.58, 0.03))
	ax_noise = plt.axes((0.1, 0.15, 0.58, 0.03))
	ax_gain_slope = plt.axes((0.1, 0.1, 0.58, 0.03))
	ax_clear = plt.axes((0.81, 0.42, 0.17, 0.07))

	slider_constant = Slider(ax_constant, "Constant", -5.0, 5.0, valinit=defaults["constant"])
	slider_noise = Slider(ax_noise, "Noise std", 0.0, 2.0, valinit=defaults["noise_std"])
	slider_gain_slope = Slider(ax_gain_slope, "Gain slope", -1.0, 1.0, valinit=defaults["gain_slope"])

	clear_button = Button(ax_clear, "Clear / Reset")

	def recompute() -> None:
		constant_value = round(slider_constant.val,2)
		noise_std = round(slider_noise.val,2)
		gain_slope = round(slider_gain_slope.val,2)

		base = np.zeros_like(x) 

		
		ax.set_title(f"Mode: signal over time ({constant_value} + {gain_slope} * x + N(0, {noise_std}))")

		noisy_signal = constant_value + x*gain_slope + rng.normal(0.0, noise_std, size=x.size)

		signal_line.set_ydata(noisy_signal)
		baseline_line.set_ydata(base)

		y_min = min(np.min(noisy_signal), np.min(base))
		y_max = max(np.max(noisy_signal), np.max(base))
		pad = max(0.5, 0.1 * (y_max - y_min if y_max != y_min else 1.0))
		ax.set_ylim(min(y_min - pad,-5), max(y_max + pad,5))

		fig.canvas.draw_idle()

	def on_slider_change(_value: float) -> None:
		recompute()


	def on_clear(_event) -> None:
		slider_constant.reset()
		slider_noise.reset()
		slider_gain_slope.reset()

		recompute()

	slider_constant.on_changed(on_slider_change)
	slider_noise.on_changed(on_slider_change)
	slider_gain_slope.on_changed(on_slider_change)
	clear_button.on_clicked(on_clear)

	recompute()
	plt.show()


if __name__ == "__main__":
	main()
