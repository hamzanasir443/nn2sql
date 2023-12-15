import duckdb
import numpy as np
from datetime import datetime
from duckdb.typing import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import psutil
import time
import logging

# Set up logging
logging.basicConfig(filename='mnist_duckdb.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')

pdf_memory_path = 'memory_usage_report.pdf'
pdf_memory = PdfPages(pdf_memory_path)

rep=1
limit=6000
sizes=[200,20]
attss=[20]
learningrate=0.01

db_file_path = 'mnist.duckdb'

# Establish connection with DuckDB using file-based storage
con = duckdb.connect(db_file_path)

createschema = '''
drop table if exists img;
drop table if exists one_hot;
drop table if exists mnist;
drop table if exists mnist2;
create table if not exists img (i int, j int, v float);
create table if not exists one_hot(i int, j int, v int, dummy int);'''

loadmnist = 'create table mnist (label float'
loadmnist2 = 'create table mnist2 (id int, label float'
loadmnistrel = '''
copy mnist from './mnist_train.csv' delimiter ',' HEADER CSV;
insert into mnist2 (select row_number() over (), * from mnist limit {});
insert into one_hot(select n.i, n.j, coalesce(i.v,0), i.v from (select id,label+1 as species,1 as v from mnist2) i right outer join (select a.a as i, b.b as j from (select generate_series as a from generate_series(1,{})) a, (select generate_series as b from generate_series(1,10)) b) n on n.i=i.id and n.j=i.species order by i,j);
'''.format(limit,limit)
for i in range(1,785):
	loadmnist += ', pixel{} float'.format(i)
	loadmnist2 += ', pixel{} float'.format(i)
	loadmnistrel += 'insert into img (select id,{},pixel{}/255 from mnist2); '.format(i,i)
loadmnist += ');'
loadmnist2 += ');'

weights ='''
drop table if exists w_xh;
drop table if exists w_ho;
create table if not exists w_xh (i int, j int, v float);
create table if not exists w_ho (i int, j int, v float);
insert into w_xh (select i.*,j.*,random()*2-1 from generate_series(1,{}) i, generate_series(1,{}) j);
insert into w_ho (select i.*,j.*,random()*2-1 from generate_series(1,{}) i, generate_series(1,{}) j);'''

train ='''with recursive w (iter,id,i,j,v) as (
  (select 0,0,* from w_xh union select 0,1,* from w_ho)
  union all
  (
  with w_now as (
     SELECT * from w
  ), a_xh(i,j,v) as (
     SELECT m.i, n.j, 1/(1+exp(-SUM (m.v*n.v)))
     FROM img AS m INNER JOIN w_now AS n ON m.j=n.i
     WHERE n.id=0 and n.iter=(select max(iter) from w_now) -- w_xh
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
justprint='''SELECT DISTINCT iter FROM w;'''
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
labelmax='SELECT max(precision) FROM (' +  label + ')'

def monitor_memory_usage(interval=1, duration=60):
    memory_usage = []
    for _ in range(int(duration / interval)):
        memory_info = psutil.virtual_memory()
        memory_usage.append(memory_info.used / (1024 * 1024))  # Convert to MB
        time.sleep(interval)
    return memory_usage

def plot_memory_usage(memory_usage, atts, limit, iterations, learning_rate, pdf_memory):
    print(f"Plotting memory usage for atts={atts}, limit={limit}, iterations={iterations}, learning_rate={learning_rate}")
    print(f"Memory usage data: {memory_usage}")
    if memory_usage:  # Check if there is data to plot
        plt.figure(figsize=(10, 6))
        plt.plot(memory_usage, label=f'Memory Usage - Atts: {atts}, Limit: {limit}, Iterations: {iterations}, LR: {learning_rate}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory Usage (MB)')
        plt.title(f'Memory Usage Over Time')
        plt.legend()
        plt.grid(True)
        pdf_memory.savefig()  # Save the current figure to the PDF
        plt.close()

def benchmark(atts, limit, iterations, learning_rate, pdf_memory):
    try:
        print(f"Starting benchmark: atts={atts}, limit={limit}, iterations={iterations}, learning_rate={learning_rate}")
        logging.info("Starting benchmark")
        duckdb.sql(createschema)
        duckdb.sql(loadmnist)
        duckdb.sql(loadmnist2)
        duckdb.sql(loadmnistrel)
        duckdb.sql(weights.format(784, atts, atts, 10))
        start = datetime.now()
        for i in range(rep):
            memory_usage = monitor_memory_usage(interval=1, duration=60)
            result = duckdb.sql(train.format(learning_rate, iterations) + labelmax).fetchall()
            plot_memory_usage(memory_usage, atts, limit, iterations, learning_rate, pdf_memory)

        time_elapsed = (datetime.now() - start).total_seconds() / rep
        accuracy = result[0][0] if result else None
        print(f"DuckDB-SQL-92,{atts},{limit},{learning_rate},{iterations},{time_elapsed},{accuracy}")
        logging.info("Benchmark completed successfully")
        return accuracy
    except Exception as e:
        logging.error(f"Error in benchmark: {e}")
        return None
# Initialize accuracies and labels lists before the benchmark loop
accuracies = []
labels = []

# Run benchmarks and collect accuracies
for atts in attss:
    for size in sizes:
        iterations = int(60 / size)
        logging.info(f"Running for atts: {atts}, size: {size}, iterations: {iterations}")
        acc = benchmark(atts, size, iterations, learningrate, pdf_memory)
        if acc is not None:
            accuracies.append(acc)
            labels.append(f'atts: {atts}, size: {size}')
        else:
            logging.info(f"No accuracy returned for atts: {atts}, size: {size}, iterations: {iterations}")

# Before closing the PDF, check if any plots were added
if pdf_memory.get_pagecount() > 0:
    pdf_memory.close()
else:
    print("No plots were added to the PDF. The PDF file will be empty.")
    pdf_memory.close()

# Check the lengths of accuracies and labels
print(f"Number of accuracies: {len(accuracies)}, Number of labels: {len(labels)}")

# Plotting the box plot for accuracies
if accuracies and labels:
    plt.figure(figsize=(10, 6))
    plt.boxplot(accuracies, labels=labels)
    plt.xlabel('Configuration')
    plt.ylabel('Accuracy')
    plt.title('Box Plot of Accuracies for Different Configurations')
    plt.grid(True)
    plt.savefig('accuracy_boxplot.pdf')
    plt.close()
else:
    print("No accuracies or labels to plot.")