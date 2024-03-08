import pymysql
import re
"""
What would you use for this task, please write your detailed answer with exact solution? 

first  I create a DB named export_domains for simulate the situation then I import sample data in table named products
second I connect my DB and get colmns both of device typr and stats_acces_Link frome products    
third I create a regex patern then I check for all domain to exist in patern if exist then ı matched and append both of domain and device type in a dict named domains_by_device  
last  I shut the connect and output the pure domains
I used MySQL(lib:pymysql) Database to keep data in  
I used librariy "re" for make regex pattern 
I was checking all character of link rows with my own created alphabet array and for loop 
but I got stuck and then I find the library and learn quickly 
so I  implied on my code to optimze it  
and I used some AI tools to make difficult(for Q6 ) sql queeries or optimize the code
"""
# DB establish
connection = pymysql.connect(host='localhost',
                             user='mert',
                             password='mert007metin',
                             database='export_domains',
                             cursorclass=pymysql.cursors.DictCursor)

try:
    with connection.cursor() as cursor:
        # get data from db.products
        cursor.execute("SELECT Device_Type, Stats_Access_Link FROM products")
        rows = cursor.fetchall()
        
        # intialize a dict to use 
        domains_by_device = {}
        
        # create a regex pattern
        domain_pattern = re.compile(r'https?://([^<]+)', re.IGNORECASE)
        
        # iteration for all row
        for row in rows:
            device_type = row['Device_Type']
            access_link = row['Stats_Access_Link']
            
            # check if domain exist in pattern
            match = domain_pattern.search(access_link)
            if match:
                domain = match.group(1)
                
                # if exist then append data on dict
                if device_type not in domains_by_device:
                    domains_by_device[device_type] = []
                domains_by_device[device_type].append(domain)
finally:
    connection.close()

# output the pure domains
for device_type, domains in domains_by_device.items():
    print("device : ",device_type)
    print("Domains :", domains)
