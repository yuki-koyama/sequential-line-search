uniform sampler2D texture; 
uniform vec3 first;
uniform vec3 second;

float rgb2h(vec3 rgb)
{
    float r = rgb[0];
    float g = rgb[1];
    float b = rgb[2];

    float M = max(r, max(g, b));
    float m = min(r, min(g, b));

    float h;
    if (M == m) {
        h = 0.0;
    } else if (m == b) {
        h = 60.0 * (g - r) / (M - m) + 60.0;
    } else if (m == r) {
        h = 60.0 * (b - g) / (M - m) + 180.0;
    } else if (m == g) {
        h = 60.0 * (r - b) / (M - m) + 300.0;
    } else {
        h = 0.0;
    }
    h /= 360.0;
    if (h < 0.0) {
        h = h + 1.0;
    } else if (h > 1.0) {
        h = h - 1.0;
    }
    return h;    
}

float rgb2s4hsv(vec3 rgb)
{
    float r = rgb[0];
    float g = rgb[1];
    float b = rgb[2];
    float M = max(r, max(g, b));
    float m = min(r, min(g, b));

    if (M < 1e-10) return 0.0;
    return (M - m) / M;
}

float rgb2s4hsl(vec3 rgb)
{
    float r = rgb[0];
    float g = rgb[1];
    float b = rgb[2];
    float M = max(r, max(g, b));
    float m = min(r, min(g, b));

    if (M - m < 1e-10) return 0.0;
    return (M - m) / (1.0 - abs(M + m - 1.0));
}

float rgb2L(vec3 rgb)
{
    float m = min(min(rgb.x, rgb.y), rgb.z);
    float M = max(max(rgb.x, rgb.y), rgb.z);

    return (M + m) * 0.5;
}

vec3 rgb2hsl(vec3 rgb)
{
    vec3 hsl;

    hsl.x = rgb2h(rgb);
    hsl.y = rgb2s4hsl(rgb);
    hsl.z = rgb2L(rgb);

    return hsl;
}

float h2rgb(float f1, float f2, float hue)
{
    if (hue < 0.0)
        hue += 1.0;
    else if (hue >= 1.0)
        hue -= 1.0;
    float res;
    if ((6.0 * hue) < 1.0)
        res = f1 + (f2 - f1) * 6.0 * hue;
    else if ((2.0 * hue) < 1.0)
        res = f2;
    else if ((3.0 * hue) < 2.0)
        res = f1 + (f2 - f1) * ((2.0 / 3.0) - hue) * 6.0;
    else
        res = f1;
    return res;
}

vec3 hsl2rgb(vec3 hsl)
{
    vec3 rgb;

    if (hsl.y == 0.0) {
        rgb = vec3(hsl.z, hsl.z, hsl.z); // Luminance
    } else {
        float f2;

        if (hsl.z < 0.5) {
            f2 = hsl.z * (1.0 + hsl.y);
        } else {
            f2 = (hsl.z + hsl.y) - (hsl.y * hsl.z);
        }

        float f1 = 2.0 * hsl.z - f2;

        rgb.x = h2rgb(f1, f2, hsl.x + (1.0 / 3.0));
        rgb.y = h2rgb(f1, f2, hsl.x);
        rgb.z = h2rgb(f1, f2, hsl.x - (1.0 / 3.0));
    }

    return rgb;
}

vec3 changeColorBalance(vec3 rgb, vec3 param)
{
    float lightness = rgb2L(rgb);

    const float a     = 0.25;
    const float b     = 0.333;
    const float scale = 0.7;

    vec3 midtones = (clamp((lightness - b) /  a + 0.5, 0.0, 1.0) * clamp((lightness + b - 1.0) / -a + 0.5, 0.0, 1.0) * scale) * param;

    vec3 newColor = rgb + midtones;
    newColor = clamp(newColor, 0.0, 1.0);

    // preserve luminosity
    vec3 newHsl = rgb2hsl(newColor);
    return hsl2rgb(vec3(newHsl.x, newHsl.y, lightness));
}

vec3 rgb2hsv(vec3 rgb)
{   
    float h = rgb2h(rgb);
    float s = rgb2s4hsv(rgb);
    float v = max(rgb.x, max(rgb.y, rgb.z));
    return vec3(h, s, v);
}

vec3 hsv2rgb(vec3 hsv) {
    vec3 rgb;

    float r, g, b, h, s, v;

    h = hsv.x;
    s = hsv.y;
    v = hsv.z;

    if (s <= 0.001) {
        r = g = b = v;
    } else {
        float f, p, q, t;
        int i;
        h *= 6.0;
        i = int(floor(h));
        f = h - float(i);
        p = v * (1.0 - s);
        q = v * (1.0 - (s * f));
        t = v * (1.0 - (s * (1.0 - f)));
        if (i == 0 || i == 6) {
            r = v; g = t; b = p;
        } else if (i == 1) {
            r = q; g = v; b = p;
        } else if (i == 2) {
            r = p; g = v; b = t;
        } else if (i == 3) {
            r = p; g = q; b = v;
        } else if (i == 4) {
            r = t; g = p; b = v;
        } else if (i == 5) {
            r = v; g = p; b = q;
        }
    }
    rgb.x = r;
    rgb.y = g;
    rgb.z = b;
    return rgb;
}

void main()
{
    vec4 color = texture2D(texture, vec2(gl_TexCoord[0]));

    vec3 p1 = first  - vec3(0.5, 0.5, 0.5);
    vec3 p2 = second - vec3(0.5, 0.5, 0.5);

    color.xyz = changeColorBalance(color.xyz, p2);

    // brightness
    color.x *= 1.0 + p1.x;
    color.y *= 1.0 + p1.x;
    color.z *= 1.0 + p1.x;

    // contrast
    float cont = tan((p1.y + 1.0) * 3.1415926535 * 0.25);
    color.x = (color.x - 0.5) * cont + 0.5;
    color.y = (color.y - 0.5) * cont + 0.5;
    color.z = (color.z - 0.5) * cont + 0.5;

    // clamp
    color = clamp(color, 0.0, 1.0);

    // saturation
    vec3 hsvVector = rgb2hsv(color.xyz);
    float s = hsvVector.y;
    s *= p1.z + 1.0;
    s = clamp(s, 0.0, 1.0);
    hsvVector.y = s;
    color.xyz = hsv2rgb(hsvVector);

    gl_FragColor = color;
}
