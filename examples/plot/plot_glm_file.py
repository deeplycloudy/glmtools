import argparse
parse_desc = """Plot a sequence of GLM data files."""

def create_parser():
    parser = argparse.ArgumentParser(description=parse_desc)
    parser.add_argument(dest='filenames',metavar='filename', nargs='*')
    parser.add_argument('-o', '--output_dir', metavar='directory',
                        required=True, dest='outdir', action='store', )
    return parser


##### END PARSING #####   
import os
import matplotlib.pyplot as plt

from glmtools.io.imagery import open_glm_time_series
 
from plots import plot_glm

fields_6panel = ['flash_extent_density', 'average_flash_area','total_energy', 
                 'group_extent_density', 'average_group_area', 'group_centroid_density']

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    glm_grids = open_glm_time_series(args.filenames)

    time_options = glm_grids.time.to_series()
    time_options.sort_values(inplace=True)

    fig = plt.figure(figsize=(18,12))
    file_tag = 'fed_afa_toe_ged_aga_gcd'
    images_out = []
    for t in time_options:
        plot_glm(fig, glm_grids, t, fields_6panel, subplots=(2,3))

        outpath = os.path.join(args.outdir, '20%s' %(t.strftime('%y/%b/%d')))
        if os.path.exists(outpath) == False:
            os.makedirs(outpath)

        # EOL field catalog convention
        eol_template = 'satellite.GOES-16.{0}.GLM_{1}.png'
        time_str = t.strftime('%Y%m%d%H%M')

        png_name = eol_template.format(time_str, file_tag)
        outfile=os.path.join(outpath, png_name)
        outfile = outfile.replace(' ', '_')
        outfile = outfile.replace(':', '')
        images_out.append(outfile)
        fig.savefig(outfile, facecolor='black', dpi=150)
    # print(images_out)