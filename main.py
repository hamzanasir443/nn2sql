import threading
from matplotlib.backends.backend_pdf import PdfPages
import duckdb
import matplotlib
import numpy as np
from datetime import datetime, time
from duckdb.typing import *
import matplotlib.pyplot as plt
import psutil
import time



rep = 1
sizes = [150, 300]
attss = [20, 50]  # Varying this parameter to observe accuracy changes
itss = [10, 100]
learningrate = 0.01
results = []

createschema = '''
drop table if exists img;
drop table if exists one_hot;
drop table if exists w_xh;
drop table if exists w_ho;
drop table if exists data;
drop table if exists data2;
create table if not exists img (i int, j int, v float);
create table if not exists one_hot(i int, j int, v int, dummy int);'''

loadiris = '''
create table if not exists data (sepal_length float,sepal_width float,petal_length float,petal_width float,species int);
create table if not exists data2 (id int, sepal_length float,sepal_width float,petal_length float,petal_width float,species int);
copy data from './iris.csv' delimiter ',' HEADER CSV;'''

loadirisrel = '''
insert into data2 (select row_number() over (), * from data);
insert into img (select id,1,sepal_length/10 from data2);
insert into img (select id,2,sepal_width/10 from data2);
insert into img (select id,3,petal_length/10 from data2);
insert into img (select id,4,petal_width/10 from data2);
insert into one_hot(select n.i, n.j, coalesce(i.v,0), i.v from (select id,species+1 as species,1 as v from data2) i right outer join (select a.a as i, b.b as j from (select generate_series as a from generate_series(1,{})) a, (select generate_series as b from generate_series(1,4)) b) n on n.i=i.id and n.j=i.species order by i,j);'''

weights ='''create table if not exists w_xh (i int, j int, v float);
create table if not exists w_ho (i int, j int, v float);
insert into w_xh (select i.*,j.*,random()*2-1 from generate_series(1,{}) i, generate_series(1,{}) j);
insert into w_ho (select i.*,j.*,random()*2-1 from generate_series(1,{}) i, generate_series(1,{}) j);'''

#SQL query to train weights
train ='''with recursive w (iter,id,i,j,v) as (
  (select 0,0,* from w_xh union select 0,1,* from w_ho)
  union all
  (
  with w_now as (
     SELECT * from w
  ), a_xh(i,j,v) as (
     SELECT m.i, n.j, 1/(1+exp(-SUM (m.v*n.v)))
     FROM img AS m INNER JOIN w_now AS n ON m.j=n.i
     WHERE m.i < {} and n.id=0 and n.iter=(select max(iter) from w_now) -- w_xh
     GROUP BY m.i, n.j
  ), a_ho(i,j,v) as (
     SELECT m.i, n.j, 1/(1+exp(-SUM (m.v*n.v))) --sig(SUM (m.v*n.v))
     FROM a_xh AS m INNER JOIN w_now AS n ON m.j=n.i
     WHERE n.id=1 and n.iter=(select max(iter) from w_now)  -- w_ho
     GROUP BY m.i, n.j
  ), l_ho(i,j,v) as (
     select m.i, m.j, 2*(m.v-n.v)
     from a_ho AS m INNER JOIN one_hot AS n ON m.i=n.i AND m.j=n.j
  ), d_ho(i,j,v) as (
     select m.i, m.j, m.v*n.v*(1-n.v)
     from l_ho AS m INNER JOIN a_ho AS n ON m.i=n.i AND m.j=n.j
  ), l_xh(i,j,v) as (
     SELECT m.i, n.i as j, (SUM (m.v*n.v)) -- transpose
     FROM d_ho AS m INNER JOIN w_now AS n ON m.j=n.j
     WHERE n.id=1 and n.iter=(select max(iter) from w_now)  -- w_ho
     GROUP BY m.i, n.i
  ), d_xh(i,j,v) as (
     select m.i, m.j, m.v*n.v*(1-n.v)
     from l_xh AS m INNER JOIN a_xh AS n ON m.i=n.i AND m.j=n.j
  ), d_w(id,i,j,v) as (
     SELECT 0, m.j as i, n.j, (SUM (m.v*n.v))
     FROM img AS m INNER JOIN d_xh AS n ON m.i=n.i
     WHERE m.i < {}
     GROUP BY m.j, n.j
     union
     SELECT 1, m.j as i, n.j, (SUM (m.v*n.v))
     FROM a_xh AS m INNER JOIN d_ho AS n ON m.i=n.i
     GROUP BY m.j, n.j
  )
  select iter+1, w.id, w.i, w.j, w.v - {} * d_w.v
  from w_now as w, d_w
  where iter < {} and w.id=d_w.id and w.i=d_w.i and w.j=d_w.j
  )
)'''

#print the weights
justprint='''SELECT DISTINCT iter FROM w;'''

#predict and generate the label
label='''SELECT iter, count(*)::float/(select count(distinct i) from one_hot) as precision
FROM (
   SELECT *, rank() over (partition by m.i,iter order by v desc) as rank
   FROM (
      SELECT m.i, n.j, 1/(1+exp(-SUM (m.v*n.v))) as v, m.iter
      FROM (
         SELECT m.i, n.j, 1/(1+exp(-SUM (m.v*n.v))) as v, iter
         FROM img AS m INNER JOIN w AS n ON m.j=n.i
         WHERE n.id=0 -- and n.iter=(select max(iter) from w)
         GROUP BY m.i, n.j, iter ) AS m INNER JOIN w AS n ON m.j=n.i
      WHERE n.id=1 and n.iter=m.iter
      GROUP BY m.i, n.j, m.iter
   ) m ) pred,
   (SELECT *, rank() over (partition by m.i order by v desc) as rank FROM one_hot m) test
WHERE pred.i=test.i and pred.rank = 1 and test.rank=1
GROUP BY iter, pred.j=test.j
HAVING (pred.j=test.j)=true
ORDER BY iter
'''

#take the best prediction
labelmax='SELECT max(precision) FROM (' +  label + ')'

# Function to check memory usuage
def monitor_memory_usage(interval=1, duration=60):
    memory_usage = []
    for _ in range(int(duration / interval)):
        memory_info = psutil.virtual_memory()
        memory_usage.append(memory_info.used / (1024 * 1024))  # Convert to MB
        time.sleep(interval)
    return memory_usage #return list of all memory points

#This is generate the pdf and graphs
def plot_memory_usage(memory_usage, current_iteration, pdf_memory):
    plt.figure(figsize=(10, 6))
    plt.plot(memory_usage, label=f'Memory Usage during Iteration {current_iteration} (MB)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Memory Usage (MB)')
    plt.title(f'Memory Usage Over Time - Iteration {current_iteration}')
    plt.legend()
    plt.grid(True)
    pdf_memory.savefig()  # Save the current figure to the PDF
    plt.close()  # Close the figure to free memory

# Initializing PDF for memory graphs
pdf_memory_path = 'memory_usage_report.pdf'
pdf_memory = PdfPages(pdf_memory_path)

# Initialize a list to store accuracy data
accuracy_results = []

def benchmark(atts, limit, iterations, learning_rate, pdf_memory, iteration_number):
    duckdb.sql(createschema)
    for i in range(int(limit / 150)):
        duckdb.sql(loadiris)
    duckdb.sql(loadirisrel.format(limit))
    duckdb.sql(weights.format(4, atts, atts, 3))

    loadtime = datetime.now()
    start = datetime.now()
    for i in range(rep):
        memory_usage = monitor_memory_usage(interval=1, duration=60)
        result = duckdb.sql(train.format(limit, limit, learning_rate, iterations) + labelmax).fetchall()
        plot_memory_usage(memory_usage, iteration_number, pdf_memory)

    time = (datetime.now() - start).total_seconds() / rep
    accuracy = result[0][0] if result else None
    accuracy_results.append({'atts': atts, 'limit': limit, 'iterations': iterations, 'accuracy': accuracy})
    print("DuckDB-SQL-92,{},{},{},{},{},{}".format(atts, limit, learning_rate, iterations, time, accuracy))


iteration_counter = 1
for atts in attss:
    for size in sizes:
        for iterations in itss:
            benchmark(atts, size, iterations, learningrate, pdf_memory, iteration_counter)
            iteration_counter += 1

# Close the PDF for memory graphs
pdf_memory.close()

accuracy_by_atts = {}
for result in accuracy_results:
    atts = result['atts']
    accuracy = result['accuracy']
    if atts not in accuracy_by_atts:
        accuracy_by_atts[atts] = []
    accuracy_by_atts[atts].append(accuracy)

# Prepare data for box plot
data_to_plot = [accuracies for accuracies in accuracy_by_atts.values()]
labels = [str(atts) for atts in accuracy_by_atts.keys()]

# Create and display the box plot
plt.figure(figsize=(10, 6))
plt.boxplot(data_to_plot, labels=labels)
plt.xlabel('Number of Attributes')
plt.ylabel('Accuracy')
plt.title('Box Plot of Accuracies for Different Number of Attributes')
plt.grid(True)
plt.show()