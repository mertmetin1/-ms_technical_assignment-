import pymysql.cursors

# conn establish
connection = pymysql.connect(host='localhost',
                             user='mert',
                             password='mert007metin',
                             database='country_vacation_stats_db',
                             cursorclass=pymysql.cursors.DictCursor)

try:
    with connection.cursor() as cursor:
        # querry
        sql = """
        SELECT 
            c.country,
            c.date,
            IFNULL(cv.daily_vaccinations, 0) AS daily_vaccinations,
            c.vaccines
        FROM 
            (SELECT DISTINCT country, date, vaccines FROM country_vacation_stats_table) AS c
        LEFT JOIN 
            country_vacation_stats_table AS cv ON c.country = cv.country AND c.date = cv.date
        LEFT JOIN 
            (SELECT 
                country,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY daily_vaccinations) AS median_daily_vaccinations
            FROM 
                country_vacation_stats_table
            WHERE 
                daily_vaccinations IS NOT NULL
            GROUP BY 
                country) AS median_table ON c.country = median_table.country
        ORDER BY 
            c.country, c.date;
        """

        # querry exec
        cursor.execute(sql)

        # save changes
        connection.commit()



finally:
    #shut the conn
    connection.close()
