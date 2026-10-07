"""Paste into a Script CHOP callbacks DAT after the README's venv setup.

Set TouchDesigner's external Python module path to .venv-td/Lib/site-packages.
Reference: https://derivative.ca/UserGuide/Python#Installing_Custom_Python_Packages
"""

from stereo_surround.touchdesigner import TouchDesignerAdapter

# Change settings here, then press Reset. Music example:
# SETTINGS = dict(listening_mode="music", center_width=0.35, rear_gain=0.7)
SETTINGS = dict(listening_mode="standard")

def onSetupParameters(scriptOp):
    page = scriptOp.appendCustomPage('Surround')
    page.appendPulse('Reset', label='Reset decoder')


def onPulse(par):
    if par.name == 'Reset':
        par.owner.store('surround_adapter', TouchDesignerAdapter(**SETTINGS))


def onCook(scriptOp):
    adapter = scriptOp.fetch('surround_adapter', None, search=False)
    if adapter is None:
        adapter = TouchDesignerAdapter(**SETTINGS)
        scriptOp.store('surround_adapter', adapter)
    adapter.cook(scriptOp.inputs[0], scriptOp)
