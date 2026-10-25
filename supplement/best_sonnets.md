# Sonnet subset

A subset of five sonnets per configuration, out of the 90 samples of each (3 seeds x 30). All samples are in `results/samples/`. Output is lowercase without punctuation, as the model generates it.

Columns: rhyme pairs = target-scheme pairs whose rhyme suffixes match by sound, with repeated end words counted as misses; repeats = repeated line-final words; meter = verses with 10-12 syllables (our counter) and verses Rantanplan scans as exactly 11; perplexity under Salamandra-7B and Gemma-4-12B (lower is better); Gemma-4-31B-it judge, 1-5 for grammaticality / coherence / poetic quality. In parentheses: percentile within the 90 samples of the configuration (100 = best).

Reference values (all samples): human sonnets 49 / 47 perplexity, judge 4.50 / 4.78 / 4.34; Claude Haiku 4.5 21 / 20, judge 3.67 / 4.60 / 2.90.

## K = 2 (target ABBA ABBA CDE CDE)

Configuration mean over 90 samples: rhyme 69.5%, repeats 1.38, meter 8.7/14 (10-12) and 4.6/14 (Rantanplan = 11), perplexity 154 / 158, judge 1.62 / 1.68 / 1.80.

### 1. seed 302, sample 15

```
en vano que en el alma tercero
no está el todo el dolor arrebatado
y muerda el aura regalado
en despojos de tu curso primero
es el encanto en el agua severo
aunque sonámbula el rigor le mojado
es la dichoso que cuanto olvidado
su risa mortal en silencio espero
mas para decretos no acabado
estas amoroso que cuida
pues en no más lo más más debe
feliz tan espíritu que mi cuidado
que es dolor adoré a la vida
alegría en sus ruiseñor la muerte
```

| rhyme pairs | repeats | meter 10-12 | Rantanplan = 11 | ppl Salamandra | ppl Gemma | judge |
|---|---|---|---|---|---|---|
| 14/15 | 0 | 10/14 | 9/14 | 121 (81) | 110 (85) | 2 / 2 / 2 |

### 2. seed 302, sample 25

```
que es fuego de amor nace el encendido
no forma en la esperanza vida
que de un corazón furor rendida
y no de hallo vive tal seguido
si este siendo de siempre rendido
por cuanto sentimos con sola salida
será qué quita mi homicida
contento en cada remacha de mi sentido
no sé pues bien no que en los ojos
de polvo los pies y a el pecho
la blanca muerte que infusas los despojos
no me encanto amor ni mis enojos
lo que pecado es vivir dichoso
ni ha de muerte que me ojos
```

| rhyme pairs | repeats | meter 10-12 | Rantanplan = 11 | ppl Salamandra | ppl Gemma | judge |
|---|---|---|---|---|---|---|
| 14/15 | 1 | 9/14 | 1/14 | 141 (58) | 110 (88) | 2 / 2 / 2 |

### 3. seed 202, sample 19

```
si por risa y el amor muero
y quiero a contento y no será enajenado
que amor el dichoso de mi más pecado
que el rendido muero que me quiero
si es esto de amor la fe espero
en mi alma que el mundo y ha pasado
de la misma acreditan ser vendado
y en vos por lo que de mí he quiero
amar el bien cuando me dila
y el mal a este que me escondida
que más queréis de amar feliz cansado
y pasare de razón amor me trasladaron
y quiero como llegaste de mi vida
y si me suya el azul lo estado
```

| rhyme pairs | repeats | meter 10-12 | Rantanplan = 11 | ppl Salamandra | ppl Gemma | judge |
|---|---|---|---|---|---|---|
| 13/15 | 1 | 8/14 | 2/14 | 95 (94) | 93 (99) | 2 / 2 / 2 |

### 4. seed 102, sample 23

```
rinde el fuego que perseverar del viento
y el remedio resultan el cielo el cielo
laca el risa del carro velo
gritas el abismo a su pensamiento
feliz su tierra el pesados aliento
que faras en ave su vestido vuelo
cisma el silencio luces el suelo
a su rigor humilde a nace el viento
ya no pudo libre cabello y encanto
por tu londres el castalio espanto
que el sol ay lo que mí más tanto
mas el alma llama que llanto
dichoso está azul bañarse por tanto
que todo lo entregué la lleva su manto
```

| rhyme pairs | repeats | meter 10-12 | Rantanplan = 11 | ppl Salamandra | ppl Gemma | judge |
|---|---|---|---|---|---|---|
| 14/15 | 2 | 11/14 | 7/14 | 209 (8) | 151 (54) | 1 / 2 / 2 |

### 5. seed 302, sample 11

```
dulce alegre aunque hermosa fiera
soberbia humilla bebieron más constante
rayos espíritu primera arado
que en tu turbados queda modera
si no celestial amor si doquiera
la vez de la aurora el bastante
la vista más rayo suya honrado
y en aminora a su honra verdadera
oh justa abrazado mas en vano
el feliz porque el universo tesoro
tanto desdén que a su nombre cercano
y en vano belleza que tenéis mano
que fuera hijo que aun en oro
sazón el famoso de tan humano
```

| rhyme pairs | repeats | meter 10-12 | Rantanplan = 11 | ppl Salamandra | ppl Gemma | judge |
|---|---|---|---|---|---|---|
| 11/15 | 0 | 11/14 | 11/14 | 174 (26) | 223 (7) | 1 / 1 / 2 |

## K = 5 (target ABBA ABBA CDE CDE)

Configuration mean over 90 samples: rhyme 82.0%, repeats 2.31, meter 8.8/14 (10-12) and 5.1/14 (Rantanplan = 11), perplexity 127 / 125, judge 1.54 / 1.70 / 1.81.

### 1. seed 102, sample 23

```
no el fuego amor que el sentimiento
y el remedio mi bien en vuestra pena
dichoso mi a mi año ajena
que en mi locura mi pensamiento
alegría mi daño el mi aliento
que fuera mi deseo y la vena
infernales que escrito aunque más enfrena
que no ha bien pues nace el tormento
ya en risa libre gerarchía es vana
en lo cubre giro en gusto sigo
y vos el fin y lo que mí la alcanza
mas tiene un bien que me engaña
que me ave me desasida no es castigo
que todo lo mismo mi mal la esperanza
```

| rhyme pairs | repeats | meter 10-12 | Rantanplan = 11 | ppl Salamandra | ppl Gemma | judge |
|---|---|---|---|---|---|---|
| 15/15 | 0 | 10/14 | 10/14 | 120 (51) | 102 (74) | 2 / 2 / 2 |

### 2. seed 302, sample 25

```
si es fuego de amor feliz el fuego
desasida la memoria de la vida
que el tiempo amor ración escondida
y el deleite vive de azul luego
si el dichoso de la región el fuego
risa de aprenden conocida
que por quien dárdano la homicida
que mal remacha de belleza ciego
qué fuera ninguno amor y es suena
que inextinto el pecado y el pensamiento
la razón que i el sentido
no sé blancor amor ni lo condena
que que quiero tener es entendimiento
que es sonando encanto más desceñido
```

| rhyme pairs | repeats | meter 10-12 | Rantanplan = 11 | ppl Salamandra | ppl Gemma | judge |
|---|---|---|---|---|---|---|
| 14/15 | 1 | 12/14 | 10/14 | 152 (21) | 129 (37) | 2 / 2 / 2 |

### 3. seed 302, sample 9

```
soy el fuego me halla por el amor
y en el mar y el oro la tierra mía
yo la luz entre el sol de día
y en el feliz entre la noche y el amor
pero en la intolerable del miz color
infelices la noche de mi alegría sombría
los pordiosero dichoso a mi alegría
que detenerse de la noche a mi amador
la azul del cielo mi quise cielo
que como mi llanto vence me surtidor
en el alma en la noche mi hielo
en el triste noche de mi consuelo
y como el viento de contento perlas
y en el aire mi amor en el cielo
```

| rhyme pairs | repeats | meter 10-12 | Rantanplan = 11 | ppl Salamandra | ppl Gemma | judge |
|---|---|---|---|---|---|---|
| 13/15 | 2 | 12/14 | 1/14 | 51 (100) | 60 (99) | 2 / 2 / 2 |

### 4. seed 202, sample 2

```
muerte negación del semblante velo
en el rayo a la noche el día
del mundo en la alegría mía
la virgen objeto el valle y del cielo
el noche contento y el celo
lucha de florentín en mi día
y está divino el azul fría
que por constante de las puertas vuelo
así no entre la tierra y fiera
dichoso fiesta de la grandeza
en las pertinaz feliz comido
al fin mi bien que placer fiera
que el sol de las almas estrecheza
que del sol de luz rayo de olvido
```

| rhyme pairs | repeats | meter 10-12 | Rantanplan = 11 | ppl Salamandra | ppl Gemma | judge |
|---|---|---|---|---|---|---|
| 13/15 | 2 | 8/14 | 9/14 | 105 (69) | 104 (73) | 2 / 2 / 2 |

### 5. seed 302, sample 11

```
dulce error qué desdén qué río
cuando escamas el imposible cuidado
feliz esta resucitó tan mezclado
que mal abrasado encanto el río
dulce abstinente amor el dulce frío
me cumplidos en vano el cuidado
la contento el mal en el honrado
y sólo embotará a mi dolor mío
oh azul amor y el pensamiento
que el deleite y el corazón tanto
tanto quedó que a mi llanto tanto
deje en vano a mi mi tormento
que parecían el gloria que en tanto
pudiera el pecado de mi llanto
```

| rhyme pairs | repeats | meter 10-12 | Rantanplan = 11 | ppl Salamandra | ppl Gemma | judge |
|---|---|---|---|---|---|---|
| 12/15 | 4 | 11/14 | 11/14 | 94 (82) | 77 (93) | 2 / 1 / 2 |

## K = 5 + cross-group repulsion (target ABBA ABBA CDE CDE)

Configuration mean over 90 samples: rhyme 78.4%, repeats 1.50, meter 9.1/14 (10-12) and 5.1/14 (Rantanplan = 11), perplexity 131 / 131, judge 1.59 / 1.71 / 1.79.

### 1. seed 102, sample 11

```
cierto comprimida y dichoso contento
con que en celos si a mi deseado
encanto el pecho y vida cansado
de risa de mi muerte el tormento
dejando mi vida señor tan viento
y el deseo de mi salado
tan volvédmelas tan enjugar la sitiado
que flecha el rigor a mi sentimiento
pero si de aquel fin suspendido
del ave a la vida es homicida
que en la razón de mi bien fuerte
quede es la fe de mi olvido
pues basta la vida que la vida
y moderada la muerte de la muerte
```

| rhyme pairs | repeats | meter 10-12 | Rantanplan = 11 | ppl Salamandra | ppl Gemma | judge |
|---|---|---|---|---|---|---|
| 15/15 | 0 | 10/14 | 7/14 | 75 (96) | 84 (90) | 2 / 2 / 2 |

### 2. seed 102, sample 23

```
no es fuego amor que el sentimiento
y el remedio mi bien me han pasado
que mi esperanza mi tan desengañado
que en el siringa mi pensamiento
alegría mi daño el mi aliento
que fuera mi fuerza que amor pesado
por el dichoso aunque más enamorado
que no es remedio que nace el tormento
ya no risa razón no es presente
en lo cubre algas en su gusto
y dolor el fin y lo que mí la alcanza
mas tiene polo feliz que me tridente
que me ave me restituyes no es justo
que todo lo mismo mi mal la esperanza
```

| rhyme pairs | repeats | meter 10-12 | Rantanplan = 11 | ppl Salamandra | ppl Gemma | judge |
|---|---|---|---|---|---|---|
| 15/15 | 0 | 12/14 | 8/14 | 101 (75) | 86 (88) | 2 / 2 / 2 |

### 3. seed 202, sample 2

```
placer negación del semblante velo
en el cuello a la noche el día
del día en la alegría mía
la reposada noche el sol y el cielo
gules sí contento con el velo
el sol de oro en la día
la noche grandísima el azul fría
el dichoso que de sus virtudes vuelo
así bravos balada de colores
que la negra novedad de los enojos
en los hileras feliz fingido
al fin pues y que hora bullidores
que el hijo de sus alemana ojos
que del sol que deleite de olvido
```

| rhyme pairs | repeats | meter 10-12 | Rantanplan = 11 | ppl Salamandra | ppl Gemma | judge |
|---|---|---|---|---|---|---|
| 13/15 | 2 | 11/14 | 9/14 | 99 (80) | 100 (79) | 2 / 2 / 2 |

### 4. seed 302, sample 22

```
si ya de amor vida advierte
de la muerte de pecado de la vida
tan tengo comprime de mi vida
hasta la asegurarme la pasión suerte
si el magandina de la verdad fuerte
ninguno de la muerte medida
ultratumba tiene la vida de mi vida
que gloria me docto de la muerte
amor vivo de la muerte y fulgurante
que risa de su verdad tiranía
que en la muerte mal que el alma alcanza
dichoso tiene a ti y no constante
que he de morir para alegría
que siempre el orfandad de gran venganza
```

| rhyme pairs | repeats | meter 10-12 | Rantanplan = 11 | ppl Salamandra | ppl Gemma | judge |
|---|---|---|---|---|---|---|
| 12/15 | 2 | 12/14 | 6/14 | 96 (84) | 75 (94) | 2 / 2 / 2 |

### 5. seed 302, sample 7

```
cuando la vida encanece la pena
con tanta pasos de la alma mía
que a estremecida de amor la dolor cría
rechazo la niega de mi cadena
que no se aviene de mi pena
que con su intento pecado y alegría
y dichoso la razón que porfía
y tener la desterró el placer ajena
no recata sepultura su mal traerá
que es la mudanza de mi vida
que menos de morir con razón fuerte
si es tu misma la vida traerá
que no es ajena el mal que vida
se premiada el dolor de la muerte
```

| rhyme pairs | repeats | meter 10-12 | Rantanplan = 11 | ppl Salamandra | ppl Gemma | judge |
|---|---|---|---|---|---|---|
| 12/15 | 3 | 10/14 | 9/14 | 67 (99) | 63 (100) | 2 / 2 / 2 |

## K = 5, CDC DCD target (target ABBA ABBA CDC DCD)

Configuration mean over 90 samples: rhyme 80.2%, repeats 2.98, meter 9.2/14 (10-12) and 5.3/14 (Rantanplan = 11), perplexity 115 / 114, judge 1.50 / 1.68 / 1.80.

### 1. seed 302, sample 17

```
militar en la cansas soñada
gozo entramos el pecho lustroso
que tórrida en el alma dichoso
que semblante del fuego coronada
el referencia presumís feliz pensada
que de cantar de su dolor victorioso
que la mérito de lomos celoso
y el temor de placer se espada
el dulce nido a la edad desvía
donde su en dehesa juntamente
y foro dándose el orbe desconfía
no erré de la luz luciente
contento dulce realidad de la agonía
tan ardientes de su frente
```

| rhyme pairs | repeats | meter 10-12 | Rantanplan = 11 | ppl Salamandra | ppl Gemma | judge |
|---|---|---|---|---|---|---|
| 18/18 | 0 | 9/14 | 2/14 | 137 (21) | 107 (48) | 2 / 2 / 2 |

### 2. seed 302, sample 19

```
esta imposible que de la undosa
en cuya canséis dejadles y cansado
viste en imponen fuelgo y dorado
corona de la experiencia imperiosa
sola la selva humilde y rosa
y de la muerte vida apretado
y alegría que la envidia el hado
tuvisteis de amor la envidia rosa
no la suerte que risa el vencido
explora su vida tan fuerte
cuando fomente los cielos olvido
y intrépida feliz placer el advierte
y ea de amor su perseguido
que es el contento dichoso muerte
```

| rhyme pairs | repeats | meter 10-12 | Rantanplan = 11 | ppl Salamandra | ppl Gemma | judge |
|---|---|---|---|---|---|---|
| 17/18 | 1 | 11/14 | 11/14 | 122 (31) | 119 (35) | 1 / 1 / 2 |

### 3. seed 302, sample 20

```
rabie el dolor tanta pensamiento
pinte la ninfa de mi vida
umbrosa extendidas la dichoso perdida
de tu mí del corazón el viento
imposible es de ti imposible el viento
y la razón de tu fin caída
y el contento de la noche perdida
la rosa ayer que en noche aliento
imposible el amor gentil de la mía
y escuchas se das el otro ausente
que que es mi amor la gozo fría
seguro en manos de feliz gente
y el atención de tu cristal falsía
como belleza a la iniquidad frente
```

| rhyme pairs | repeats | meter 10-12 | Rantanplan = 11 | ppl Salamandra | ppl Gemma | judge |
|---|---|---|---|---|---|---|
| 16/18 | 2 | 9/14 | 9/14 | 137 (20) | 124 (28) | 2 / 2 / 2 |

### 4. seed 302, sample 11

```
dulce señor que en he tenido
ambición cuando alegría más cuidado
porque cantar risa mezclado
que en esta arraiga ceda desceñido
dulce benigno amor que más sido
me retenerte en vano el cuidado
de contento el duro que el honrado
y sólo brutos a su causa suspendido
oh cruel amor dejándose anhelo
que si tan feliz su engaño tanto
dichoso feliz que a su muerte vuelo
no en vano de mi padecer tanto
que parecían el gloria que al cielo
pudiera el llanto de mi llanto
```

| rhyme pairs | repeats | meter 10-12 | Rantanplan = 11 | ppl Salamandra | ppl Gemma | judge |
|---|---|---|---|---|---|---|
| 16/18 | 2 | 11/14 | 10/14 | 95 (65) | 90 (71) | 2 / 2 / 2 |

### 5. seed 302, sample 15

```
en oficio que amor en el mío
y advertencia los ojos de mis ojos
y insultes mis mis antojos
en los despojos de este amor mío
vacilaba la causa y el hielo desvarío
de imposible de rigor de mis ojos
pues cuando dichoso en los antojos
las incrusta testa del claro río
abrevió no moverá no es venganza
que a un fiero estado airado
en que en mi bien apena alcanza
feliz tan muero placer mi cuidado
que es que no gozo mi esperanza
tener en ojos risa el estado
```

| rhyme pairs | repeats | meter 10-12 | Rantanplan = 11 | ppl Salamandra | ppl Gemma | judge |
|---|---|---|---|---|---|---|
| 15/18 | 3 | 9/14 | 12/14 | 108 (48) | 115 (39) | 2 / 2 / 2 |

Note: within each seed, configurations start from the same noise, so the same sample index can give related poems across configurations (for example seed 102, sample 23 under K = 5 and K = 5 + repulsion share most of the octave).
