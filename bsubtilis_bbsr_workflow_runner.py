from inferelator_ng.bbsr_tfa_workflow import BBSR_TFA_Workflow
from inferelator_ng import utils

utils.Debug.set_verbose_level(utils.Debug.levels['verbose'])

workflow = BBSR_TFA_Workflow()
# Common configuration parameters
workflow.input_dir = 'data/bsubtilis'
workflow.num_bootstraps = 2
workflow.delTmax = 110
workflow.delTmin = 0
workflow.tau = 45
workflow.run() 
