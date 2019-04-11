Adding a new grid
=================


Key places that define the grids produced:

--- SETUP ---
- grid.make_grids: GLMGridder
in pipeline_setup, self.outgrids holds the arrays for each grid type
in output_setup, names of the fields and other metadata for output

- grid.make_grids: GLMlutGridder (inherits from above, overrides F/G/E pipeline and output_setup)
event_pipeline_setup
group_pipeline_setup
flash_pipeline_setup
Each of these sets up a pipeline that receives data from mimic_lma.fast_fixed_grid_read_flash_chunk (*.send lines) and routes it to the
target grid (the accumulate_*_on_grid segments). Set up in reverse order.


--- PROCESSING ---
- io.mimic_lma.fast_fixed_grid_read_flash_chunk:
Calls split_event_dataset_from_props in grid.split_events to get event geometry info (but no physics), followed by io.mimic_lma. replicate_and_weight_split_child_dataset_new, which uses consolidated event data from io.glm.get_lutevents (i.e., flash_data) and the geometry from the previous step to produce a final dataset of weighted values of interest for each event shard corresponding to one fixed grid quad. That dataset is then mimic_lma_dataset_lut which calls _fake_lma_events_from_split_glm_lutevents to copy the replicated/wedighted data into the data structure expected by the F/G/E pipeline. See also _fake_lma_from_glm_lutgroups, _fake_lma_from_glm_lutflashes


- io.glm.get_lutevents: yield line and dtype will need to be changed to add any new parameters of interest as a pre-summary step.


--- OUTPUT ---
Depends on master or unifiedgridfile
Will need support for new fields in grid plotting routines.


Implementation
==============

I'll break this down into three steps:
1. Adding a field and saving it empty
2. Adding a new pipeline to replicate an existing calculation
3. Adding a new calculation and routing that to the new pipeline

A test environment
------------------
As part of development I need to run the same data over and over again. I will use the notebook `glm_test_data_new_grid_dev.ipynb` to do so.


Adding a new field
------------------
Commit:


Adding a new pipeline
---------------------
Commit:
Additional post-processing step needed to replicate average area after accumulation. Have to divide by FED. We won't need this for minimum flash area,
but it's good to see where that calculation takes place for average flash area.
Background: dabeaz coroutines

Adding a new calculation to feed the pipeline.

Don't forget to remove the post-processing step needed above.