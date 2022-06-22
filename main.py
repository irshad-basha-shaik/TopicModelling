
#shubham topic modelling
@app.route('/topic_modelling')
def topic_modelling():
    return render_template('topic_modelling/layout.html')


@app.route('/topic_modelling/home')
def topic_modelling_home():
    return render_template('topic_modelling/home.html')


@app.route('/topic_modelling/doc_rules')
def topic_modelling_docs_rules():
    return render_template('topic_modelling/doc_rules.html')


@app.route('/topic_modelling/generate_concept')
def topic_modelling_generate_concept():
    return render_template('topic_modelling/generate_concept.html')


@app.route('/topic_modelling/generate_thesarus')
def topic_modelling_generate_thesarus():
    return render_template('topic_modelling/generate_thesarus.html')



@app.route('/topic_modelling/generate_results')
def topic_modelling_generate_generate_results():
    return render_template('topic_modelling/generate_results.html')

@app.route('/topic_modelling/project_list')
def topic_modelling_project_list():
    return render_template('topic_modelling/project_list.html')

