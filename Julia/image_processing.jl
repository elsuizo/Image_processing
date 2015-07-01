#= -------------------------------------------------------------------------
# @file image_processing.jl
#
# @date 07/01/15 17:51:05
# @author Martin Noblia
# @email martin.noblia@openmailbox.org
#
# @brief
# Images processing functions in Julia language
# @detail
#
  Licence:
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at
# your option) any later version.
# 
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public License

---------------------------------------------------------------------------=#
module ImageProcessing

using Images, Color

function chromatics_coord(img::AbstractArray)

    x, y = size(img)
    a = zeros(img)
    for i in 1:x
        for j in 1:y
            r = img[i, j].r 
            g = img[i, j].g 
            b = img[i, j].b
            s = r + g + b
            a[i, j] = RGB(r / s, g / s, b / s)
        end
    end
    return a 
end

function white_patch(img::AbstractArray)
    x, y = size(img) 
    out = zeros(img)
    r_m = maximum(red(img))
    g_m = maximum(green(img))
    b_m = maximum(blue(img))
    
    for i in 1:x
        for j in 1:y
            r = img[i, j].r 
            g = img[i, j].g 
            b = img[i, j].b
            out[i, j] = RGB(r / r_m, g / g_m, b / b_m)
        end
    end
    return out
end

function histogram(img)
    
    _, counts = hist(img[:], -1/256:1/256:1)

    s₁,s₂ = size(img_gray)
    return counts / (s₁ * s₂)
    
end

function eq_hist(img_gray)
    
    h = histogram(img_gray) # calculate the histogram
    h_eq = cumsum(h) # calculate the cumulative sum
    gr = [Gray(i) for i in h_eq] # generated the equalized grayscale map
    A = float(data(img_gray)) # convert to float
    dataint = iround(Uint8, 254*A + 1 ) # convert 1-254
    img_eq = ImageCmap(dataint, gr)
    
    return img_eq
end


function fc(img)
    
    # LUT colors
    red = zeros(256)
    green = zeros(256)
    blue = zeros(256)
    
    # red LUT
    red[1:43] = 1.0
    red[43:85] = linspace(1, 0, 43)
    red[172:214] = linspace(0, 1, 43)
    red[214:end] = 1.0
    red = red * RGB(1, 0, 0)
    # green LUT
    green[1:43] = linspace(0, 1, 43)
    green[43:129] = 1.0
    green[129:171] = linspace(1, 0, 43)
    green = green * RGB(0, 1, 0)
    # blue LUT
    blue[86:128] = linspace(0, 1, 43)
    blue[129:214] = 1.0
    blue[214:end] = linspace(1, 0, 43)
    blue = blue * RGB(0, 0, 1)
    
    A = float(data(img)) # convert to float
    dataint = iround(Uint8, 254*A + 1 ) # convert
    false_color = red + green + blue # blend the colors
    img_fc = ImageCmap(dataint, false_color)
    
    return img_fc
end


function segment_Gonz(img_gray)
    T = 0.5 * (minimum(img_gray) + maximum(img_gray)) 
    flag = false
    while ~flag
        g = img_gray .< T
        T_next = 0.5 * (mean(img_gray[g]) + mean(img_gray[~g]) )
        flag = abs(T - T_next) < 0.5
        T = T_next
    end
    return T
end

end
