# Third-party notices

Gaussian-JEPA builds on and modifies components from the following projects.
Their original notices and license terms continue to apply to those components.

## ShapeSplat / Gaussian-MAE

- Project: <https://github.com/qimaqi/ShapeSplat-Gaussian_MAE>
- Authors: Qi Ma et al.
- License: Creative Commons Attribution-ShareAlike 4.0 International
- Use here: Gaussian data loading, grouping, transformer infrastructure, and
  downstream evaluation scaffolding, modified for Gaussian-JEPA.

## Point-MAE

- Project: <https://github.com/Pang-Yatian/Point-MAE>
- Copyright (c) 2022 PANG-Yatian, YUAN-Li
- License: MIT
- Use here: point-cloud transformer and downstream training infrastructure,
  subsequently adapted for Gaussian inputs.

The MIT license requires the following notice to accompany substantial
portions of the software:

> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to
> deal in the Software without restriction, including without limitation the
> rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
> sell copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in
> all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

## PyTorch3D

`utils/rotation_conversions.py` is adapted from
[PyTorch3D](https://github.com/facebookresearch/pytorch3d), Copyright (c) Meta
Platforms, Inc. and affiliates. It retains the upstream copyright header and
is covered by PyTorch3D's BSD-style license.

This repository is distributed under CC BY-SA 4.0, as stated in the root
`LICENSE` file. Third-party components retain the upstream notices and terms
identified above.
