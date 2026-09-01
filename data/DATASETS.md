# Datasets

Directorio para gestión de datasets del proyecto.

## Dataset seleccionado: CelebA

### Información general

- **Nombre**: CelebA (CelebFaces Attributes Dataset)
- **Fuente**: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- **Licencia**: Solo para investigación
- **Tamaño**: ~200,000 imágenes de celebridades
- **Anotaciones**: 40 atributos faciales binarios

### Atributos incluidos

El dataset CelebA incluye los siguientes 40 atributos faciales:

1. Atr_in_back
2. Atr_in_front
3. Atr_5_o_clock_shadow
4. Atr_arched_eyebrows
5. Atr_bags_under_eyes
6. Atr_bald
7. Atr_bangs
8. Atr_big_lips
9. Atr_big_nose
10. Atr_black_hair
11. Atr_blond_hair
12. Atr_blurry
13. Atr_brown_hair
14. Atr_bushy_eyebrows
15. Atr_chubby
16. Atr_double_chin
17. Atr_eyeglasses
18. Atr_goatee
19. Atr_gray_hair
20. Atr_heavy_makeup
21. Atr_high_cheekbones
22. Atr_male
23. Atr_mouth_slightly_open
24. Atr_mustache
25. Atr_narrow_eyes
26. Atr_no_beard
27. Atr_oval_face
28. Atr_pale_skin
29. Atr_pointy_nose
30. Atr_receding_hairline
31. Atr_rosy_cheeks
32. Atr_sideburns
33. Atr_smiling
34. Atr_straight_hair
35. Atr_wavy_hair
36. Atr_wearing_earrings
37. Atr_wearing_hat
38. Atr_wearing_lipstick
39. Atr_wearing_necklace
40. Atr_wearing_necktie

### Atributos visualmente observables (selección para el proyecto)

Para este proyecto, nos enfocaremos en atributos **visualmente observables**:

- **Accesorios**: eyeglasses, hat, earrings, necklace, necktie, lipstick
- **Cabello**: black_hair, blond_hair, brown_hair, gray_hair, bangs, receding_hairline, straight_hair, wavy_hair, bald
- **Rostro**: smiling, mouth_slightly_open, narrow_eyes, big_lips, big_nose, oval_face, pointy_nose, high_cheekbones, rosy_cheeks, chubby, double_chin, pale_skin
- **Facial hair**: goatee, mustache, no_beard, sideburns
- **Otros**: 5_o_clock_shadow, arched_eyebrows, bags_under_eyes, bushy_eyebrows, blurry, heavy_makeup

### Estructura del dataset

```
data/raw/
├── images/
│   ├── img_000001.jpg
│   ├── img_000002.jpg
│   └── ...
└── annotations/
    └── attributes.csv
```

### Formato de anotaciones

CSV con las siguientes columnas:
- `image_id`: Identificador único de la imagen
- `Atr_eyeglasses`: 0 o 1
- `Atr_hat`: 0 o 1
- ... (cada atributo)

### Descarga

El dataset debe descargarse manualmente desde la página oficial y colocarse en `data/raw/`.

**Nota**: El dataset CelebA tiene licencia solo para investigación. No se puede usar con fines comerciales.

### Referencias

- Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep Learning Face Attributes in the Wild. In Proceedings of the IEEE International Conference on Computer Vision (ICCV).
