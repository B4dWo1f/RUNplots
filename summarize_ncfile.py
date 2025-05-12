#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import sys
from netCDF4 import Dataset

def summarize_ncfile(nc_path, output_txt=None):
   ncfile = Dataset(nc_path, 'r')

   lines = []

   # Dimensions
   lines.append("=== Dimensions ===")
   for dimname, dim in ncfile.dimensions.items():
      lines.append(f"Dimension: {dimname}, Size: {len(dim)}")

   # Variables
   lines.append("\n=== Variables ===")
   for varname, var in ncfile.variables.items():
      lines.append(f"Variable: {varname}")
      lines.append(f"  Dimensions: {var.dimensions}")
      lines.append(f"  Shape: {var.shape}")
      for attrname in var.ncattrs():
         attrval = var.getncattr(attrname)
         lines.append(f"    {attrname} = {attrval}")

   # Global Attributes
   lines.append("\n=== Global Attributes ===")
   for attrname in ncfile.ncattrs():
      attrval = ncfile.getncattr(attrname)
      lines.append(f"{attrname} = {attrval}")

   ncfile.close()

   summary = "\n".join(lines)
   print(summary)

   # Save to file if requested
   if output_txt:
      with open(output_txt, "w") as f:
          f.write(summary)


if __name__ == '__main__':
   if len(sys.argv) < 2:
      msg  = 'Error: File not specified.\n'
      msg += 'Usage:  python summarize_ncfile.py <filename>'
      print(msg)
      sys.exit(1)
   
   fname = sys.argv[1]
   summarize_ncfile(fname)
