conn_str = (
            'DRIVER={SQL Server};'
            'SERVER=140.136.149.181,1433;' 
            'DATABASE=news;'  
            'UID=news;'  
            'PWD=123;' 
        )
 CREATE TABLE news_table (
    Id INT PRIMARY KEY IDENTITY(1,1),  -- 自動遞增的主鍵
    Category NVARCHAR(255) NOT NULL,    -- 文章類別
    Subcategory NVARCHAR(255),          -- 文章子類別
    Title NVARCHAR(255) NOT NULL,       -- 文章標題
    Abstract NVARCHAR(MAX),             -- 文章摘要
    Content NVARCHAR(MAX),              -- 文章內容
    URL NVARCHAR(255),                  -- 文章網址
    PublicationDate DATE                -- 發佈日期
);
