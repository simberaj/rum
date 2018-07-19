def dictToTable(data, heads, rowHead=False):
    if isinstance(data, list):
        data = {key : [str(item[key]) for item in data] for key in data[0]}
    else:
        data = {key : [str(value)] for key, value in data.items()}
    rows = []
    for key, name in heads:
        rows.append([name] + data[key])
    return listToTable(rows, colHead=True, rowHead=rowHead)
    
def listToTable(rows, colHead=False, rowHead=False):
    htmlRows = []
    htmlRows = [rowToHTML(rows[0], 'th')] if rowHead else []
    for row in rows[(1 if rowHead else 0):]:
        htmlRows.append(rowToHTML(row, 'td', 'th' if colHead else None))
    return '<table><tbody><tr>' + '</tr><tr>'.join(htmlRows) + '</tr></tbody></table>'
    
def rowToHTML(row, tag, firstTag=None):
    if firstTag:
        return startTag(firstTag) + row[0] + endTag(firstTag) + rowToHTML(row[1:], tag)
    else:
        start = startTag(tag)
        end = endTag(tag)
        return start + (end + start).join(row) + end
        
def startTag(tag):
    return '<' + tag + '>'
    
def endTag(tag):
    return '</' + tag + '>'