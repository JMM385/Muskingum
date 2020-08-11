#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Librerías
from os.path import join, dirname
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
# BOKEH
from bokeh.io import curdoc, output_notebook, output_file, show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Slider, HoverTool, CDSView, IndexFilter


# In[2]:


df = pd.read_csv(join(dirname(__file__), 'data/datos_ej2.csv'))
df.head(15)


# In[3]:


plt.style.use('ggplot')


# In[4]:


fig, ax = plt.subplots()

ax.plot(df['t'], df['I'], label='Hidrograma aguas arriba')
ax.plot(df['t'], df['O'], label='Hidrograma aguas abajo')

ax.set_xlabel('t [hs]')
ax.set_ylabel('Q [m$^3$/s]')

ax.legend(loc='upper right')
plt.show();


# In[5]:


delta_t = 4.0 # Los datos deben estar uniformemente espaciados. En este caso c/ 4hs.
X = 0.10 # X toma valores entre 0 y 0.5, propongo valor inicial

# Defino los términos numerador y denominador del método gráfico
def numerador(I_i, I_ii, O_i, O_ii):
    
    num = 0.5 * delta_t * ((I_ii + I_i) - (O_i + O_ii))
    
    return num

def denominador(I_i, I_ii, O_i, O_ii):
    
    den = X * (I_ii - I_i) + (1 - X)*(O_ii - O_i)
    
    return den


# In[6]:


# Defino df_2 igual a df sin las primeras 10 filas (justificado en que debe transcurrir un cierto tiempo hasta que la onda del hidrograma llegue)
df_2 = df.loc[9:].reset_index()
df_2.head()


# In[7]:


# Aplico la función numerador y denominador 
for j in range(len(df_2)-1):
    
    a = df_2.loc[j, 'I']
    b = df_2.loc[j+1, 'I']
    c = df_2.loc[j, 'O']
    d = df_2.loc[j+1, 'O']
    
    df_2.loc[j,'num'] = numerador(a, b, c, d)
    df_2.loc[j, 'den'] = denominador(a, b, c, d)
    

df_2.head(10)


# In[8]:


fig, ax = plt.subplots()

ax.scatter(df_2['den'], df_2['num'])

ax.set_xlabel('Denominador')
ax.set_ylabel('Numerador')

plt.show();


# In[9]:


# Preparo df_2 para hacer ajuste por cuadrados mínimos
# Elimino última fila (contiene NAs)
df_2 = df_2[: -1]
df_2.head()


# In[10]:


x_train = df_2['den'].values.reshape(-1, 1)
y_train = df_2['num'].values.reshape(-1, 1)
reg = LinearRegression(fit_intercept=False) #Iniciar regresión

reg.fit(x_train, y_train) #Fit the regressor
prediction_space = np.linspace(min(x_train), max(x_train), num=len(df_2)).reshape(-1, 1) # Create space where to predict
y_pred = reg.predict(prediction_space) #Predict

df_2['x_test'] = prediction_space
df_2['y_test'] = y_pred

df_2.tail()


# In[11]:


k_01 = float(reg.coef_)
print('La pendiente [K(hs)]:',  k_01, '\n',
'La ordenada al origen:', reg.intercept_,  '\n',
'R^2:' , reg.score(df_2['den'].values.reshape(-1, 1), df_2['num'].values.reshape(-1, 1))
     )


# In[12]:


plt.scatter(df_2['den'].values, df_2['num'].values, color='blue')
plt.plot(prediction_space, reg.predict(prediction_space), color='black', linewidth=2.5)

plt.show();


# In[13]:


# Cálculo de R^2 y K para todo Xi

x_range = np.arange(0.0, 0.51, 0.01) # Xi para los cuales se calcularan los parámetros R^2 y K

def denominador_X(X, I_i, I_ii, O_i, O_ii):
    """Calcula el término denominador para la serie de caudales I y O para X dado"""
    den = X * (I_ii - I_i) + (1 - X)*(O_ii - O_i)
    
    return den

df_dens = pd.DataFrame({
                        'I' : df.loc[9 :, 'I'],
                        'O' : df.loc[9 :, 'O']
                       }) # df para guardar los 50 denominadores a calcular
df_dens = df_dens.reset_index()

for _ in x_range:
    # Aplico la función denominador 
    for j in range(len(df_dens)-1):

        a = df_dens.loc[j, 'I']
        b = df_dens.loc[j+1, 'I']
        c = df_dens.loc[j, 'O']
        d = df_dens.loc[j+1, 'O']
 
        df_dens.loc[j, 'den_' + str(_)] = denominador_X(_, a, b, c, d)


# In[14]:


# Verificación
df_dens['den_0.1'].head()


# In[15]:


df_dens = df_dens[: -1]
df_dens = df_dens.drop(columns=['index', 'I', 'O'])
df_dens.tail()


# In[16]:


# Ajuste por cuadrados mínimos para cada Xi
reg_2 = LinearRegression(fit_intercept=False)
#reg_2.fit(x_train, y_train)
r2_i = []
k_i = []
for i, column in df_dens.items():
    x_train = column.values.reshape(-1, 1)
    reg_2.fit(x_train, y_train)
    r2_i.append(reg_2.score(x_train, y_train)) # R^2 para cada Xi
    k_i.append(float(reg_2.coef_)) # K para cada Xi


# In[17]:


df_xi = pd.DataFrame({'x' : x_range,
                      'r2' : r2_i,
                      'k' : k_i
                    })
df_xi.head(15)


# In[18]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.plot(df_xi['x'], df_xi['r2'], c='red')
ax1.scatter(df_xi['x'], df_xi['r2'], c='black')

ax1.set_title('R$^2$ en función de X')
ax1.set_xlabel('X')
ax1.set_ylabel('R$^2$')

ax2.plot(df_xi['x'], df_xi['k'], c='red')
ax2.scatter(df_xi['x'], df_xi['k'], c='black')
ax2.set_title('K [hr] en función de X')
ax2.set_xlabel('X')
ax2.set_ylabel('K [hr]')

fig.tight_layout()
plt.show();


# In[19]:


def ruteo(x, k, I_ii, I_i, O_i):
    """Ruteo del hidrograma aguas arriba para x y k asociado. Devuelve O_ii """
    c1 = (delta_t - 2*k*x) / (2*k*(1-x) + delta_t)
    
    c2 = (delta_t + 2*k*x) / (2*k*(1-x) + delta_t)
    
    c3 = (2*k*(1-x) - delta_t) / (2*k*(1-x) + delta_t)
    
    O_ii = c1*I_ii + c2*I_i + c3*O_i
    
    return O_ii


# In[20]:


# Cálculo de O_modelado para x y k
df['O_modelado'] = np.nan
df.loc[0, 'O_modelado'] = df.loc[0, 'I'] #el valor inicial del O_modelado se toma igual al inicial observado aguas arriba

for o in range(1, len(df)):
    df.loc[o, 'O_modelado'] = ruteo(X, k_01, df.loc[o, 'I'], df.loc[o-1, 'I'], df.loc[o-1, 'O_modelado'] )

df.head()


# In[21]:


fig, ax = plt.subplots()

ax.plot(df['t'], df['I'], label='Hidrograma aguas arriba observado')
ax.plot(df['t'], df['O'], label='Hidrograma aguas abajo observado')
ax.plot(df['t'], df['O_modelado'], label='Hidrograma aguas abajo modelado')

ax.set_xlabel('t [hs]')
ax.set_ylabel('Q [m$^3$/s]')

ax.legend(loc='upper right')
plt.show();


# In[22]:


# Creo slider para X tomando valores entre 0 y 0.5, c/0.01

slider_X = Slider(start=0.0, end=0.5, step=0.01, value=0.25, title='X')

# Creo ColumnDataSource                                                             
source = ColumnDataSource(data={
                                'den' : df_2['den'],
                                'num' : df_2['num'],
                                'x_test' : df_2['x_test'],
                                'y_test' : df_2['y_test'],
                               })
source_3 = ColumnDataSource(data={
                                'I' : df['I'],
                                'O' : df['O'],
                                'O_modelado' : df['O_modelado'],
                                't' : df['t']
})


# In[23]:


def callback(attr, old, new):
    
    X = slider_X.value
    def denominador(I_i, I_ii, O_i, O_ii):
    
        den = X * (I_ii - I_i) + (1 - X)*(O_ii - O_i)

        return den

    for j in range(len(df_2)-1):
    
        a = df_2.loc[j, 'I']
        b = df_2.loc[j+1, 'I']
        c = df_2.loc[j, 'O']
        d = df_2.loc[j+1, 'O']

        df_2.loc[j, 'den'] = denominador(a, b, c, d)
    
    # Ajuste LOS
    x_t = df_2['den'].values.reshape(-1, 1)
    y_t = df_2['num'].values.reshape(-1, 1)
    reg = LinearRegression(fit_intercept=False) #Iniciar regresión
    reg.fit(x_t, y_t) #Fit the regressor
    prediction_space = np.linspace(min(x_t), max(x_t), num=len(df_2)).reshape(-1, 1) # Create space where to predict
    y_pred = reg.predict(prediction_space) #Predict
    df_2['x_test'] = prediction_space
    df_2['y_test'] = y_pred
    
    # O_modelado
    k_x = float(reg.coef_)
    for o in range(1, len(df)):
        df.loc[o, 'O_modelado'] = ruteo(X, k_x, df.loc[o, 'I'], df.loc[o-1, 'I'], df.loc[o-1, 'O_modelado'] )
    
    new_data_1 = {
                'den' : df_2['den'],
                'num' : df_2['num'],
                'x_test' : df_2['x_test'],
                'y_test' : df_2['y_test'],
                               }
    new_data_3 = {
                    'I' : df['I'],
                    'O' : df['O'],
                    'O_modelado' : df['O_modelado'],
                    't' : df['t']
        
    }
    
    source.data = new_data_1
    source_3.data = new_data_3

slider_X.on_change('value', callback)


# In[24]:


# Figura 1 Numerador vs Denominador variando X
p_1 = figure(x_axis_label='Numerador',
           y_axis_label='Denominador',
           title='Método Gráfico',
           plot_width=600, plot_height=400,
           x_range=(-4, 16), y_range=(-300, 1200)
          )
p_1.background_fill_color = "beige"
p_1.background_fill_alpha = 0.75

p_1.circle(x='den', y='num', source=source, color='black', fill_color='purple', fill_alpha=0.3, size=7.5)
p_1.line(x='x_test', y='y_test', source=source, color='black', legend_label='Ajuste por cuadrados mínimos')
p_1.legend.location = 'bottom_right'

# Figura 2 Hidrogramas aguas arriba y abajo
p_2 = figure(x_axis_label='t [hs]',
             y_axis_label='Q [m^3/s]',
             title='Hidrogramas',
             plot_width=600, plot_height=400
            )
p_2.background_fill_color = "beige"
p_2.background_fill_alpha = 0.75

p_2.line(x='t', y='I', source=source_3, legend_label='Hidrograma aguas arriba observado', color='blue')
p_2.line(x='t', y='O', source=source_3, legend_label='Hidrograma aguas abajo observado', color='orange')

p_2.line(x='t', y='O_modelado', source=source_3, legend_label='Hidrograma aguas abajo modelado', color='black')

p_2.varea(x='t', y1=0, y2='O', source=source_3, alpha=0.15, fill_color='orange')
p_2.varea(x='t', y1=0, y2='I', source=source_3, alpha=0.15, fill_color='blue')


# In[32]:


# Figura 5 Observado vs Modelado variando X
p_5 = figure(x_axis_label='O modelado [m^3/s]',
           y_axis_label='O observado [m^3/s]',
           title='Comparación O_observado vs O_modelado',
           plot_width=600, plot_height=400,
           x_range=(-75, 250), y_range=(-75, 250)
          )
p_5.background_fill_color = "beige"
p_5.background_fill_alpha = 0.75

p_5.circle(x='O_modelado', y='O', source=source_3, color='black', fill_color='green', fill_alpha=0.3, size=7.5)
p_5.line(x=[i for i in range(-75, 250)], y=[i for i in range(-75, 250)], legend_label='Y=X')
p_5.legend.location = 'top_left'


# In[26]:


source_2 = ColumnDataSource(data={'x' : df_xi['x'],
                                  'r2' : df_xi['r2'],
                                  'k' : df_xi['k']})

# Hover Tool
hover_r = [('X', '$x'), ('R^2', '$y{0.0000}')]
hover_k = [('X', '$x'), ('K', '$y')]

# Filter solution: (X, K) by index
view = CDSView(source=source_2, filters=[IndexFilter([14])]) # la solución se encuentra en la fila 14


# In[27]:


# Figura 3 R^2 en función de X
p_3 = figure(x_axis_label='X',
             y_axis_label='R^2',
             title='R^2 en función de X',
             plot_width=600, plot_height=400,
             tooltips=hover_r
            )
p_3.background_fill_color = "beige"
p_3.background_fill_alpha = 0.75

p_3.circle(x='x', y='r2', source=source_2, color='black', size=3, fill_alpha=0.75)
p_3.circle(x='x', y='r2', source=source_2, view=view, color='red', size=8, legend_label='X/ R^2 es máximo') # Highlight MAX

p_3.line(x='x', y='r2', source=source_2, color='green')

# Figura 4 K en función de X
p_4 = figure(x_axis_label='X',
             y_axis_label='K [hs]',
             title='K en función de X',
             plot_width=600, plot_height=400,
             tooltips=hover_k
            )
p_4.background_fill_color = "beige"
p_4.background_fill_alpha = 0.75



p_4.circle(x='x', y='k', source=source_2, color='black', size=3, fill_alpha=0.75)
p_4.circle(x='x', y='k', source=source_2, view=view, color='red', size=8, legend_label='K/ R^2 es máximo') # Highlight K
p_4.legend.location = 'top_left'
p_4.line(x='x', y='k', source=source_2, color='green')


# In[28]:


from bokeh.layouts import row, column


# In[29]:


layout = row(slider_X, column(p_1, p_3),column(p_2, p_4), p_5)
curdoc().add_root(layout)

