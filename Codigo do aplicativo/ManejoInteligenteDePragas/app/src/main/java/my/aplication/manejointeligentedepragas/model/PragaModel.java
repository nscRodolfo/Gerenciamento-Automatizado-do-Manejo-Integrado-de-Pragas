package my.aplication.manejointeligentedepragas.model;

public class PragaModel {
    private int Cod_Praga;
    private String Nome;
    private String Familia;
    private String Ordem;
    private String Descricao;
    private String NomeCientifico;
    private String Localizacao;
    private String AmbientePropicio;
    private String CicloVida;
    private String Injurias;
    private String Observacoes;
    private String HorarioDeAtuacao;
    private String EstagioDeAtuacao;
    private String ControleCultural;
    private int Status;

    public PragaModel(int cod_Praga, String nome, String familia, String ordem, String descricao, String nomeCientifico, String localizacao, String ambientePropicio, String cicloVida, String injurias, String observacoes, String horarioDeAtuacao, String estagioDeAtuacao, int Status, String controleCultural) {
        this.Cod_Praga = cod_Praga;
        this.Nome = nome;
        this.Familia = familia;
        this.Ordem = ordem;
        this.Descricao = descricao;
        this.NomeCientifico = nomeCientifico;
        this.Localizacao = localizacao;
        this.AmbientePropicio = ambientePropicio;
        this.CicloVida = cicloVida;
        this.Injurias = injurias;
        this.Observacoes = observacoes;
        this.HorarioDeAtuacao = horarioDeAtuacao;
        this.EstagioDeAtuacao = estagioDeAtuacao;
        this.Status = Status;
        this.ControleCultural = controleCultural;
    }
    public PragaModel(String nome) {
        this.Nome = nome;
    }

    public PragaModel(){
        this.Cod_Praga = 0;
        this.Nome = null;
        this.Familia = null;
        this.Ordem = null;
        this.Descricao = null;
        this.NomeCientifico = null;
        this.Localizacao = null;
        this.AmbientePropicio = null;
        this.CicloVida = null;
        this.Injurias = null;
        this.Observacoes = null;
        this.HorarioDeAtuacao = null;
        this.EstagioDeAtuacao = null;
        this.Status = 1;
        this.ControleCultural = null;
    }

    public int getCod_Praga() {
        return Cod_Praga;
    }

    public void setCod_Praga(int cod_Praga) {
        this.Cod_Praga = cod_Praga;
    }

    public String getNome() {
        return Nome;
    }

    public void setNome(String nome) {
        this.Nome = nome;
    }

    public String getFamilia() {
        return Familia;
    }

    public void setFamilia(String familia) {
        this.Familia = familia;
    }

    public String getOrdem() {
        return Ordem;
    }

    public void setOrdem(String ordem) {
        this.Ordem = ordem;
    }

    public String getDescricao() {
        return Descricao;
    }

    public void setDescricao(String descricao) {
        this.Descricao = descricao;
    }

    public String getNomeCientifico() {
        return NomeCientifico;
    }

    public void setNomeCientifico(String nomeCientifico) {
        this.NomeCientifico = nomeCientifico;
    }

    public String getLocalizacao() {
        return Localizacao;
    }

    public void setLocalizacao(String localizacao) {
        this.Localizacao = localizacao;
    }

    public String getAmbientePropicio() {
        return AmbientePropicio;
    }

    public void setAmbientePropicio(String ambientePropicio) {
        this.AmbientePropicio = ambientePropicio;
    }

    public String getCicloVida() {
        return CicloVida;
    }

    public void setCicloVida(String cicloVida) {
        this.CicloVida = cicloVida;
    }

    public String getInjurias() {
        return Injurias;
    }

    public void setInjurias(String injurias) {
        this.Injurias = injurias;
    }

    public String getObservacoes() {
        return Observacoes;
    }

    public void setObservacoes(String observacoes) {
        this.Observacoes = observacoes;
    }

    public String getHorarioDeAtuacao() {
        return HorarioDeAtuacao;
    }

    public void setHorarioDeAtuacao(String horarioDeAtuacao) {
        this.HorarioDeAtuacao = horarioDeAtuacao;
    }

    public String getEstagioDeAtuacao() {
        return EstagioDeAtuacao;
    }

    public void setEstagioDeAtuacao(String estagioDeAtuacao) {
        this.EstagioDeAtuacao = estagioDeAtuacao;
    }

    public int getStatus() {
        return Status;
    }

    public void setStatus(int status) {
        Status = status;
    }

    public String getControleCultural() {
        return ControleCultural;
    }

    public void setControleCultural(String controleCultural) {
        ControleCultural = controleCultural;
    }
}
