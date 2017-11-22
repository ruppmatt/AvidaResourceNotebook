import progressbar
import seaborn as sb
from tempfile import NamedTemporaryFile

def write_temp_file(contents, **kw):
    fn = NamedTemporaryFile(mode='w', delete=False, **kw)
    fn.write(contents)
    path = fn.name
    fn.close()
    return path

class ColorMaps:
    green = sb.cubehelix_palette(
        start=2, rot=0, hue=1, dark=0.10, light=1.0, gamma=1, n_colors=16,
        as_cmap=True)


class TimedProgressBar(progressbar.ProgressBar):

    def __init__(self, title='', **kw):

        self._pbar_widgets = [\
                        title,
                        '  ',
                        progressbar.Bar(),
                        progressbar.Percentage(),
                        '  ',
                        progressbar.ETA(
                            format_zero='%(elapsed)s elapsed',
                            format_not_started='',
                            format_finished='%(elapsed)s elapsed',
                            format_NA='',
                            format='%(eta)s remaining')]
        progressbar.ProgressBar.__init__(self, widgets=self._pbar_widgets, **kw)




class TimedCountProgressBar(progressbar.ProgressBar):

    def __init__(self, title='', **kw):
        self._pbar_widgets = [
            title,
            ' ',
            progressbar.FormatLabel('(%(value)s of %(max_value)s)'),
            ' ',
            progressbar.ETA(
                format_zero='%(elapsed)s elapsed',
                format_not_started='',
                format_finished='%(elapsed)s elapsed',
                format_NA='',
                format=''
                )
            ]
        progressbar.ProgressBar.__init__(self, widgets=self._pbar_widgets, **kw)



class TimedSpinProgessBar(progressbar.ProgressBar):

    def __init__(self, title='', **kw):
        self._pbar_widgets = [
                        title,
                        '  ',
                        progressbar.AnimatedMarker(),
                        '  ',
                        progressbar.FormatLabel(''),
                        '  ',
                        progressbar.ETA(
                            format_zero = '',
                            format_not_started='',
                            format_finished='%(elapsed)s elapsed',
                            format_NA='',
                            format='')]
        progressbar.ProgressBar.__init__(self, widgets=self._pbar_widgets, **kw)

    def finish(self,code):
        self._pbar_widgets[2] = ' ' # Delete the spinny wheel
        self._pbar_widgets[4] = self._finished_widget(code)
        progressbar.ProgressBar.finish(self)

    def _finished_widget(self, code):
        if code == 0:
            return progressbar.FormatLabel('[OK]')
        else:
            return progressbar.FormatLabel('[FAILED]')
